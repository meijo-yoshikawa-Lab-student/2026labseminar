"""server_app.py"""

# 型ヒントのためのモジュールをインポートします。
# オペレーティングシステムとのやり取り（ファイルパス操作など）のためのモジュールをインポートします。
import os

# データを構造体のように扱うためのdataclassをインポートします。
from dataclasses import dataclass
# 日時を扱うためのモジュールをインポートします。
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from collections.abc import Callable

# Excelファイルを操作するためのopenpyxlライブラリをインポートします。
import openpyxl

# PyTorchライブラリをインポートします。
import torch

# Flowerの共通モジュールから、メトリクス、コンテキスト、パラメータ変換関数などをインポートします。
# Flowerの共通モジュールから、評価結果のクラスをインポートします。
# Flowerの共通モジュールから、学習結果のクラスをインポートします。
from flwr.common import (
    Context,
    EvaluateRes,
    FitRes,
    Metrics,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

# Flowerのサーバー関連のクラスをインポートします。
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

# Flowerのサーバー側でクライアントを表現するプロキシクラスをインポートします。
from flwr.server.client_proxy import ClientProxy

# Flowerのサーバー戦略として、基本的なFedAvgをインポートします。
from flwr.server.strategy import FedAvg

# openpyxlからWorkbookクラスをインポートします。
from openpyxl import Workbook

# 同じディレクトリのtaskモジュールから、モデルや重み操作関数をインポートします。
from .task import Net, get_weights, set_weights


# Excelのセルに書き込むことができる値の型を定義します。
ExcelCellValue = Union[str, int, float, None]


def scalar_to_excel_cell(metric_name: str, value: Optional[Scalar]) -> ExcelCellValue:
    """FlowerのScalarをExcelのセルに書き込める値へ変換します。"""

    if value is None:
        return "N/A"
    if isinstance(value, bool):
        raise ValueError(f"Expected {metric_name} to be a string or number, got bool")
    if isinstance(value, (str, int, float)):
        return value
    raise ValueError(
        f"Expected {metric_name} to be a string or number, got {type(value)}"
    )


class ExistingExcelFilePolicy(Enum):
    """既存のExcelファイルがある場合の書き込み方針です。"""

    # 既存のExcelファイルに行を追加します。
    APPEND = "append"
    # 既存のExcelファイルがある場合はエラーにします。
    ERROR = "error"
    # 既存のExcelファイルを新しい内容で上書きします。
    OVERWRITE = "overwrite"


@dataclass(frozen=True)
class TrainingHistoryRow:
    """学習履歴Excelに書き込む1行分のデータです。"""

    server_round: int
    train_loss: ExcelCellValue
    eval_loss: ExcelCellValue
    accuracy: ExcelCellValue
    num_partitions_reported: ExcelCellValue
    total_train_samples_round: ExcelCellValue
    first_client_train_samples: ExcelCellValue

    def to_excel_row(self) -> Tuple[ExcelCellValue, ...]:
        """openpyxlで追記できる行データへ変換します。"""

        return (
            self.server_round,
            self.train_loss,
            self.eval_loss,
            self.accuracy,
            self.num_partitions_reported,
            self.total_train_samples_round,
            self.first_client_train_samples,
        )


class TrainingHistoryExcelLog:
    """学習履歴Excelファイルの作成と追記を担当します。"""

    # 学習履歴Excelの列名です。
    headers: Tuple[str, ...] = (
        "round",
        "train_loss",
        "eval_loss",
        "accuracy",
        "num_partitions_reported",
        "total_train_samples_round",
        "first_client_train_samples",
    )

    def __init__(self, file_path: Union[str, Path]) -> None:
        """学習履歴Excelファイルのパスを保持します。"""

        self.file_path = Path(file_path)

    def ensure_file_exists(self) -> None:
        """履歴ファイルが存在しなければ、ヘッダーだけを持つExcelを作成します。"""

        if self.file_path.exists():
            return
        self._create_workbook()

    def append(
        self,
        row: TrainingHistoryRow,
        existing_file_policy: ExistingExcelFilePolicy = ExistingExcelFilePolicy.APPEND,
    ) -> None:
        """既存ファイルの扱い方に従って、学習履歴を1行書き込みます。"""

        if self.file_path.exists():
            self._write_to_existing_file(row, existing_file_policy)
            return

        self._create_workbook(row)
        print(f"History file {self.file_path} not found, created a new one.")

    def _write_to_existing_file(
        self,
        row: TrainingHistoryRow,
        existing_file_policy: ExistingExcelFilePolicy,
    ) -> None:
        """既存のExcelファイルに対して、指定された方針で行を書き込みます。"""

        if existing_file_policy == ExistingExcelFilePolicy.ERROR:
            raise FileExistsError(f"Excel history file already exists: {self.file_path}")

        if existing_file_policy == ExistingExcelFilePolicy.OVERWRITE:
            self._create_workbook(row)
            return

        workbook = openpyxl.load_workbook(self.file_path)
        sheet = workbook.active

        if sheet is None:
            raise ValueError("Failed to load Excel sheet for training history.")

        sheet.append(row.to_excel_row())
        workbook.save(self.file_path)

    def _create_workbook(self, row: Optional[TrainingHistoryRow] = None) -> None:
        """ヘッダーと任意の1行を持つ新しいExcelファイルを作成します。"""

        workbook = Workbook()
        sheet = workbook.active

        if sheet is None:
            raise ValueError("Failed to create Excel sheet for training history.")

        sheet.append(self.headers)
        if row is not None:
            sheet.append(row.to_excel_row())
        workbook.save(self.file_path)


def aggregate_train_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """複数のクライアントから集まった訓練メトリクスを集約する関数です。"""

    # 各クライアントの加重損失を格納するリストを初期化します。
    train_losses = []
    # 全クライアントのサンプル数の合計を初期化します。
    num_examples_total = 0
    # クライアントから報告されたパーティション数を格納する変数を初期化します。
    reported_num_partitions = None
    # 1ラウンドあたりの全クライアントの訓練サンプル数の合計を初期化します。
    total_train_samples_round = 0

    # 各クライアントの結果（サンプル数とメトリクス）をループ処理します。
    for num_examples, client_metrics in metrics:
        # メトリクスに'train_loss'が含まれているか確認します。
        if "train_loss" in client_metrics:
            # サンプル数で重み付けした損失をリストに追加します。
            train_losses.append(num_examples * client_metrics["train_loss"])
            # 合計サンプル数を加算します。
            num_examples_total += num_examples
        # 'num_partitions'がメトリクスにあり、まだ設定されていない場合
        if "num_partitions" in client_metrics and reported_num_partitions is None:
            # クライアントから報告されたパーティション数を記録します。
            reported_num_partitions = client_metrics["num_partitions"]
        # メトリクスに'num_train_samples'が含まれているか確認します。
        if "num_train_samples" in client_metrics:
            # ラウンド全体の訓練サンプル数に加算します。

            # 整数かどうかチェックする
            int_cmnts = client_metrics["num_train_samples"]
            if not isinstance(int_cmnts, int):
                raise ValueError(
                    f"Expected num_train_samples to be an integer, got {type(client_metrics['num_train_samples'])}"
                )
            total_train_samples_round += int_cmnts

    # 合計サンプル数が0より大きい場合、全体の加重平均損失を計算します。
    aggregated_train_loss = (
        sum(train_losses) / num_examples_total if num_examples_total > 0 else None
    )
    # 集約結果を格納する辞書を初期化します。
    result_metrics = {}
    # 集約された訓練損失が計算できた場合
    if aggregated_train_loss is not None:
        # 結果辞書に'train_loss'を追加します。
        result_metrics["train_loss"] = aggregated_train_loss
    # パーティション数が報告された場合
    if reported_num_partitions is not None:
        # 結果辞書に'num_partitions'を追加します。
        result_metrics["num_partitions"] = reported_num_partitions
    # ラウンドの合計訓練サンプル数を結果辞書に追加します。
    result_metrics["total_train_samples_round"] = total_train_samples_round

    # 集約されたメトリクス辞書を返します。
    return result_metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """複数のクライアントから集まった評価メトリクス（正解率など）の加重平均を計算する関数です。"""

    # 各クライアントの加重正解率を格納するリストを初期化します。
    accuracies = []
    # 各クライアントの加重損失を格納するリストを初期化します。
    losses = []
    # 全クライアントのサンプル数の合計を初期化します。
    num_examples_total = 0
    # 各クライアントの結果（サンプル数とメトリクス）をループ処理します。
    for num_examples, client_metrics in metrics:
        # メトリクスに'accuracy'が含まれているか確認します。
        if "accuracy" in client_metrics:
            # サンプル数で重み付けした正解率をリストに追加します。
            accuracies.append(num_examples * client_metrics["accuracy"])
        # メトリクスに'loss'が含まれているか確認します。
        if "loss" in client_metrics:
            # サンプル数で重み付けした損失をリストに追加します。
            losses.append(num_examples * client_metrics["loss"])
        # 合計サンプル数を加算します。
        num_examples_total += num_examples
    # 合計サンプル数が0の場合
    if num_examples_total == 0:
        # 正解率0.0、損失0.0を返します。
        return {"accuracy": 0.0, "loss": 0.0}
    # 全体の加重平均正解率を計算します。
    aggregated_accuracy = sum(accuracies) / num_examples_total if accuracies else 0.0
    # 全体の加重平均損失を計算します。
    aggregated_loss = sum(losses) / num_examples_total if losses else 0.0
    # 計算結果を辞書として返します。
    return {"accuracy": aggregated_accuracy, "loss": aggregated_loss}


class CustomFedAvg(FedAvg):
    """FedAvg戦略を継承し、モデルの保存やカスタムロギング機能を追加したカスタム戦略クラスです。"""

    def __init__(
        self,
        *args,
        base_save_dir: str,
        config_log_path: str,
        fit_metrics_aggregation_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """カスタム戦略の初期化メソッドです。"""

        # 親クラス（FedAvg）の初期化メソッドを呼び出します。
        super().__init__(
            *args, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, **kwargs
        )
        # モデルやログを保存するベースディレクトリをインスタンス変数として保持します。
        self.base_save_dir = base_save_dir
        # 実行設定を保存するログファイルのパスをインスタンス変数として保持します。
        self.config_log_path = config_log_path
        # クライアントからのパーティション数を書き込んだかどうかのフラグです。
        self.num_partitions_from_client_written = False
        # 保存用ディレクトリが存在しない場合は作成します。
        os.makedirs(self.base_save_dir, exist_ok=True)
        # ログ保存ディレクトリの情報をコンソールに出力します。
        print(f"Models and logs will be saved in: {self.base_save_dir}")

        # 学習履歴を保存するExcelファイルのパスを定義します。
        self.history_log_path = os.path.join(
            self.base_save_dir, "training_history.xlsx"
        )
        # 学習履歴Excelファイルを管理するクラスを初期化します。
        self.history_excel_log = TrainingHistoryExcelLog(self.history_log_path)
        # 履歴ファイルがまだ存在しない場合に作成します。
        try:
            self.history_excel_log.ensure_file_exists()
        # ファイル作成中にエラーが発生した場合
        except Exception as e:
            # エラーメッセージを出力します。
            print(f"Error creating Excel history file {self.history_log_path}: {e}")

        # 現在のラウンドの訓練損失を保持する変数を初期化します。
        self.current_round_train_loss: Optional[float] = None
        # 現在のラウンドのパーティション数を保持する変数を初期化します。
        self.current_round_num_partitions: Optional[int] = None
        # 現在のラウンドの合計訓練サンプル数を保持する変数を初期化します。
        self.current_round_total_train_samples: Optional[int] = None
        # 現在のラウンドの最初のクライアントの訓練サンプル数を保持する変数を初期化します。
        self.current_round_first_client_train_samples: Optional[int] = None

    def aggregate_fit(
        self,
        server_round: int,  # 現在のサーバーラウンド数
        results: List[
            Tuple[ClientProxy, FitRes]
        ],  # 成功したクライアントからの結果リスト
        failures: List[
            Union[Tuple[ClientProxy, FitRes], BaseException]
        ],  # 失敗したクライアントのリスト
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """fit（学習）フェーズの結果を集約するメソッドをオーバーライドします。"""

        # 親クラスのaggregate_fitを呼び出して、基本的なパラメータとメトリクスの集約を行います。
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # パラメータが正常に集約された場合
        if parameters_aggregated is not None:
            # 集約されたパラメータをNumPy配列のリストに変換します。
            ndarrays_aggregated = parameters_to_ndarrays(parameters_aggregated)
            # 保存用の新しいモデルインスタンスを作成します。
            net_to_save = Net()
            # モデルに集約された重みを設定します。
            set_weights(net_to_save, ndarrays_aggregated)
            # このラウンドのグローバルモデルの保存パスを生成します。
            save_path = os.path.join(
                self.base_save_dir, f"global_model_round_{server_round}.pth"
            )
            # モデルの状態辞書（重み）をファイルに保存します。
            torch.save(net_to_save.state_dict(), save_path)

        # 集約されたメトリクスから訓練損失を取得し、インスタンス変数に保存します。
        tmp_ma_train_loss = metrics_aggregated.get("train_loss")
        # float | None じゃないならエラー
        if tmp_ma_train_loss is not None and not isinstance(tmp_ma_train_loss, (float, int)):
            raise ValueError(
                f"Expected train_loss to be a float or int, got {type(tmp_ma_train_loss)}"
            )
        self.current_round_train_loss = float(tmp_ma_train_loss) if tmp_ma_train_loss is not None else None
        # 集約されたメトリクスからパーティション数を取得し、インスタンス変数に保存します。

        # int | None じゃないならエラー
        tmp_ma_num_partitions = metrics_aggregated.get("num_partitions")
        if tmp_ma_num_partitions is not None and not isinstance(tmp_ma_num_partitions, int):
            raise ValueError(
                f"Expected num_partitions to be an integer, got {type(tmp_ma_num_partitions)}"
            )
        self.current_round_num_partitions = int(tmp_ma_num_partitions) if tmp_ma_num_partitions is not None else None
        
        # 集約されたメトリクスから合計訓練サンプル数を取得し、インスタンス変数に保存します。
        tmp_ma_total_train_samples = metrics_aggregated.get("total_train_samples_round")
        if tmp_ma_total_train_samples is not None and not isinstance(tmp_ma_total_train_samples, int):
            raise ValueError(
                f"Expected total_train_samples_round to be an integer, got {type(tmp_ma_total_train_samples)}"
            )
        self.current_round_total_train_samples = int(tmp_ma_total_train_samples) if tmp_ma_total_train_samples is not None else None

        # パーティション数が取得でき、かつまだファイルに書き込んでいない場合
        if (
            self.current_round_num_partitions is not None
            and not self.num_partitions_from_client_written
        ):
            # 例外処理を開始します。
            try:
                # 設定ログファイルを追加書き込みモードで開きます。
                with open(self.config_log_path, "a") as f:
                    # クライアントから報告されたパーティション数を書き込みます。
                    f.write(
                        f"num-partitions-from-client: {self.current_round_num_partitions}\n"
                    )
                # 書き込みが成功したことをコンソールに出力します。
                print(
                    f"Appended num-partitions-from-client: {self.current_round_num_partitions} to {self.config_log_path}"
                )
                # 書き込み済みフラグをTrueに設定します。
                self.num_partitions_from_client_written = True
            # ファイル書き込み中にエラーが発生した場合
            except Exception as e:
                # エラーメッセージを出力します。
                print(f"Error appending num_partitions_from_client to config log: {e}")

        # 学習結果が1つ以上ある場合
        if results:
            # 最初のクライアントの結果を取得します。
            first_client_proxy, first_client_fit_res = results[0]
            # 最初のクライアントの訓練サンプル数を取得してインスタンス変数に保存します。

            tmp_fcmtnts = first_client_fit_res.metrics.get("num_train_samples")
            if tmp_fcmtnts is not None and not isinstance(tmp_fcmtnts, int):
                raise ValueError(
                    f"Expected num_train_samples to be an integer, got {type(tmp_fcmtnts)}"
                )
            self.current_round_first_client_train_samples = int(tmp_fcmtnts) if tmp_fcmtnts is not None else None
        # 学習結果がない場合
        else:
            # 最初のクライアントのサンプル数をNoneに設定します。
            self.current_round_first_client_train_samples = None
        # 集約されたパラメータとメトリクスを返します。
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,  # 現在のサーバーラウンド数
        results: List[
            Tuple[ClientProxy, EvaluateRes]
        ],  # 成功したクライアントからの評価結果
        failures: List[
            Union[Tuple[ClientProxy, EvaluateRes], BaseException]
        ],  # 失敗したクライアントのリスト
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """evaluate（評価）フェーズの結果を集約するメソッドをオーバーライドします。"""

        # 親クラスのaggregate_evaluateを呼び出して、基本的な損失とメトリクスの集約を行います。
        aggregated_loss, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        # ログに記録するための訓練損失を準備します。なければ"N/A"とします。
        train_loss_to_log = (
            self.current_round_train_loss
            if self.current_round_train_loss is not None
            else "N/A"
        )
        # ログに記録するためのパーティション数を準備します。なければ"N/A"とします。
        num_partitions_to_log = (
            self.current_round_num_partitions
            if self.current_round_num_partitions is not None
            else "N/A"
        )
        # ログに記録するための合計訓練サンプル数を準備します。なければ"N/A"とします。
        total_train_samples_to_log = (
            self.current_round_total_train_samples
            if self.current_round_total_train_samples is not None
            else "N/A"
        )
        # ログに記録するための最初のクライアントのサンプル数を準備します。なければ"N/A"とします。
        first_client_samples_to_log = (
            self.current_round_first_client_train_samples
            if self.current_round_first_client_train_samples is not None
            else "N/A"
        )

        # 評価メトリクスが集約された場合
        if metrics_aggregated:
            # メトリクスから正解率を取得します。
            accuracy = metrics_aggregated.get("accuracy")
            # メトリクスから損失を取得します。
            eval_loss = metrics_aggregated.get("loss")
            # ログに記録するための評価損失を準備します。なければ"N/A"とします。
            eval_loss_to_log = scalar_to_excel_cell("loss", eval_loss)
            # ログに記録するための正解率を準備します。なければ"N/A"とします。
            accuracy_to_log = scalar_to_excel_cell("accuracy", accuracy)
        # 評価メトリクスがない場合
        else:
            # 評価損失を"N/A"とします。
            eval_loss_to_log = "N/A"
            # 正解率を"N/A"とします。
            accuracy_to_log = "N/A"
            # ログ記録ができない旨をコンソールに出力します。
            print(
                f"Round {server_round}: Evaluation metrics not available for logging."
            )

        # ファイルに追加する行のデータを作成します。
        history_row = TrainingHistoryRow(
            server_round=server_round,
            train_loss=train_loss_to_log,
            eval_loss=eval_loss_to_log,
            accuracy=accuracy_to_log,
            num_partitions_reported=num_partitions_to_log,
            total_train_samples_round=total_train_samples_to_log,
            first_client_train_samples=first_client_samples_to_log,
        )

        # 例外処理を開始します。
        try:
            # 学習履歴Excelに新しい行を追加します。
            self.history_excel_log.append(history_row)
        # その他のファイル書き込みエラーが発生した場合
        except Exception as e:
            # エラーメッセージを出力します。
            print(f"Error writing to Excel history file {self.history_log_path}: {e}")

        # 次のラウンドのために、ラウンド固有のメトリクスをリセットします。
        self.current_round_train_loss = None
        # パーティション数をリセットします。
        self.current_round_num_partitions = None
        # 合計訓練サンプル数をリセットします。
        self.current_round_total_train_samples = None
        # 最初のクライアントのサンプル数をリセットします。
        self.current_round_first_client_train_samples = None

        # 親クラスから受け取った集約損失とメトリクスを返します。
        return aggregated_loss, metrics_aggregated


def server_fn(context: Context):
    """サーバーを起動するためのコンポーネントを定義する関数です。"""

    # 実行コンテキストからサーバーの総ラウンド数を取得します。なければデフォルト値1を使用します。
    num_rounds = int(context.run_config.get("num-server-rounds", 1))
    # 実行コンテキストから学習に参加するクライアントの割合を取得します。なければデフォルト値0.5を使用します。
    fraction_fit = float(context.run_config.get("fraction-fit", 0.5))
    # 実行コンテキストから乱数シードを取得します。なければデフォルト値42を使用します。
    random_seed = int(context.run_config.get("random-seed", 42))
    # 実行コンテキストからローカルエポック数を取得します。なければデフォルト値1を使用します。
    local_epochs_config = int(context.run_config.get("local-epochs", 1))
    # 実行コンテキストから最小利用可能クライアント数を取得します。なければデフォルト値2を使用します。
    min_available_clients_config = int(
        context.run_config.get("min-available-clients", 2)
    )
    # 実行コンテキストから学習率を取得します。なければデフォルト値0.001を使用します。
    learning_rate_config = float(context.run_config.get("learning-rate", 0.001))

    # 初期グローバルモデルの重みを取得します。
    ndarrays = get_weights(Net())
    # NumPy配列の重みをFlowerのParameters形式に変換します。
    parameters = ndarrays_to_parameters(ndarrays)

    # 現在の日時を取得します。
    now = datetime.now()
    # 日時を "YYYYMMDD_HHMMSS" 形式の文字列に変換します。
    datetime_str = now.strftime("%Y%m%d_%H%M%S")
    # データセット名を変数に格納します。
    dataset_name = "mnist"
    # 保存用ベースディレクトリ名を生成します。
    base_save_dir = f"{dataset_name}_models_{datetime_str}"
    # 保存用ディレクトリが存在しない場合は作成します。
    os.makedirs(base_save_dir, exist_ok=True)

    # 実行設定を保存するログファイルのパスを定義します。
    config_log_path = os.path.join(base_save_dir, "run_configuration.txt")

    # 設定ログファイルを書き込みモードで開きます。
    with open(config_log_path, "w") as f:
        # データセット名を書き込みます。
        f.write(f"dataset: {dataset_name}\n")
        # サーバーラウンド数を書き込みます。
        f.write(f"num-server-rounds: {num_rounds}\n")
        # 学習参加クライアントの割合を書き込みます。
        f.write(f"fraction-fit: {fraction_fit}\n")
        # ローカルエポック数を書き込みます。
        f.write(f"local-epochs-configured: {local_epochs_config}\n")
        # 乱数シードを書き込みます。
        f.write(f"random-seed-configured: {random_seed}\n")
        # 最小利用可能クライアント数を書き込みます。
        f.write(f"min-available-clients-configured: {min_available_clients_config}\n")
        # 学習率を書き込みます。
        f.write(f"learning-rate-configured: {learning_rate_config}\n")

    # 設定が保存されたことをコンソールに出力します。
    print(f"Saved initial run configuration to {config_log_path}")

    def fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        """各ラウンドでクライアントに渡す設定を生成する関数を定義します。"""

        # クライアントに渡す設定情報を辞書で返します。
        return {
            "server_round": server_round,  # 現在のラウンド数
            "local_epochs": local_epochs_config,  # ローカルエポック数
            "learning_rate": learning_rate_config,  # 学習率
        }

    # カスタマイズしたFedAvg戦略をインスタンス化します。
    strategy = CustomFedAvg(
        base_save_dir=base_save_dir,  # 保存用ディレクトリ
        config_log_path=config_log_path,  # 設定ログのパス
        fraction_fit=fraction_fit,  # 学習に参加するクライアントの割合
        fraction_evaluate=1.0,  # 評価に参加するクライアントの割合（ここでは全クライアント）
        min_available_clients=min_available_clients_config,  # サーバーがラウンドを開始するための最小クライアント数
        min_fit_clients=max(
            1, int(min_available_clients_config * fraction_fit)
        ),  # 学習に参加する最小クライアント数
        min_evaluate_clients=min_available_clients_config,  # 評価に参加する最小クライアント数
        initial_parameters=parameters,  # 初期グローバルモデルのパラメータ
        evaluate_metrics_aggregation_fn=weighted_average,  # 評価メトリクスの集約関数
        fit_metrics_aggregation_fn=aggregate_train_metrics,  # 訓練メトリクスの集約関数
        on_fit_config_fn=fit_config_fn,  # 各学習ラウンドでクライアント設定を生成する関数
    )
    # サーバーの設定を定義します。ここでは総ラウンド数を設定します。
    config = ServerConfig(num_rounds=num_rounds)

    # サーバーアプリケーションのコンポーネント（戦略と設定）を返します。
    return ServerAppComponents(strategy=strategy, config=config)


# ServerAppを作成します。Flowerはこのappオブジェクトを使ってサーバーアプリケーションを実行します。
app = ServerApp(server_fn=server_fn)
