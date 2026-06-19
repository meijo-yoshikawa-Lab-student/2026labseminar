"""client_app.py"""

# Flowerのクライアントアプリケーション関連のクラスをインポートします。
from flwr.client import ClientApp, NumPyClient

# Flowerの共通モジュールからContextをインポートします。実行コンテキスト情報へのアクセスに使います。
from flwr.common import Context

# 同じディレクトリのtaskモジュールから必要な関数やクラスをインポートします。
from .task import (
    Net,  # ニューラルネットワークモデルのクラス
    get_default_device,  # 利用可能な計算デバイスを選ぶ関数
    get_weights,  # モデルの重みを取得する関数
    load_data,  # データを読み込む関数
    set_weights,  # モデルに重みを設定する関数
    test,  # モデルを評価する関数
    train,  # モデルを学習する関数
)


class FlowerClient(NumPyClient):
    """FlowerのNumPyClientを継承して、クライアントの具体的な振る舞いを定義するクラスです。"""

    def __init__(self, net, trainloader, valloader, local_epochs, num_partitions):
        """クライアントの初期化メソッドです。"""

        # ニューラルネットワークモデルをインスタンス変数として保持します。
        self.net = net
        # 訓練データローダーをインスタンス変数として保持します。
        self.trainloader = trainloader
        # 検証データローダーをインスタンス変数として保持します。
        self.valloader = valloader
        # ローカルでの学習エポック数をインスタンス変数として保持します。
        self.local_epochs = local_epochs
        # 全体のパーティション数をインスタンス変数として保持します。
        self.num_partitions = num_partitions
        # CUDA、Apple Metal、CPUの順に利用可能なデバイスを設定します。
        self.device = get_default_device()
        # モデルを決定したデバイスに移動します。
        self.net.to(self.device)
        # クライアントが使用しているデバイス情報を表示します。
        print(f"Client using device: {self.device}")
        # GPUアクセラレーションが利用できない場合にメッセージを表示します。
        if self.device.type == "cpu":
            # メッセージを出力します。
            print("CUDA/MPS not available, running on CPU.")

    def fit(self, parameters, config):
        """fitメソッドは、サーバーからの指示でモデルの学習を実行します。"""

        # サーバーから受信したグローバルモデルのパラメータを、自身のモデルに設定します。
        set_weights(self.net, parameters)

        # サーバーから送られてきた設定(config)から学習率を取得します。なければデフォルト値0.001を使用します。
        learning_rate = float(config.get("learning_rate", 0.001))

        # task.pyで定義したtrain関数を呼び出し、ローカルデータでモデルを学習させます。
        avg_loss = train(
            self.net,  # 学習対象のモデル
            self.trainloader,  # 訓練データローダー
            self.local_epochs,  # ローカルエポック数
            self.device,  # 使用するデバイス
            learning_rate=learning_rate,  # 学習率
        )

        # 訓練に使用したサンプル数を初期化します。
        num_train_samples = 0
        # 訓練データローダーとそのデータセットが存在するかチェックします。
        if self.trainloader and self.trainloader.dataset:
            # 訓練データセットのサンプル数を取得します。
            num_train_samples = len(self.trainloader.dataset)

        # 処理したサンプル数を設定します。
        num_examples_processed = num_train_samples

        # サーバーに返すメトリクス（指標）を辞書形式で定義します。
        metrics = {
            "train_loss": avg_loss,  # 訓練時の平均損失
            "num_partitions": self.num_partitions,  # 全体のパーティション数
            "num_train_samples": num_train_samples,  # このクライアントの訓練サンプル数
        }

        # 学習後のモデルの重み、学習に使用したサンプル数、メトリクスをタプルで返します。
        return (
            get_weights(self.net),  # 更新されたローカルモデルの重み
            num_examples_processed,  # 処理したサンプル数
            metrics,  # 訓練に関するメトリクス
        )

    def evaluate(self, parameters, config):
        """evaluateメソッドは、サーバーからの指示でモデルの評価を実行します。"""

        # サーバーから受信したグローバルモデルのパラメータを、自身のモデルに設定します。
        set_weights(self.net, parameters)
        # task.pyで定義したtest関数を呼び出し、ローカルの検証データでモデルを評価します。
        loss, accuracy = test(self.net, self.valloader, self.device)
        # 評価結果の損失、検証データセットのサンプル数、メトリクス（正解率と損失）を返します。
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "loss": loss}


def client_fn(context: Context):
    """クライアントを生成するための関数（Client Function）です。Flowerが各クライアントを初期化する際に呼び出します。"""

    # ニューラルネットワークモデルをインスタンス化します。
    net = Net()
    # 実行コンテキストから、このクライアントに割り当てられたパーティションIDを取得します。
    partition_id = context.node_config["partition-id"]
    # partition-idが整数かどうかチェック
    if not isinstance(partition_id, int):
        raise ValueError(f"Expected partition-id to be an integer, got {type(partition_id)}")
    # 実行コンテキストから、全体のパーティション数を取得します。
    num_partitions = context.node_config["num-partitions"]
    # num-partitionsが整数かどうかチェック
    if not isinstance(num_partitions, int):
        raise ValueError(f"Expected num-partitions to be an integer, got {type(num_partitions)}")

    # 実行コンテキストのrun_configから乱数シードを取得します。なければデフォルト値42を使用します。
    random_seed = int(context.run_config.get("random-seed", 42))

    # load_data関数を呼び出して、このクライアント用の訓練データと検証データを読み込みます。
    trainloader, valloader = load_data(
        partition_id,  # このクライアントのパーティションID
        num_partitions,  # 全体のパーティション数
        random_seed=random_seed,  # 乱数シード
    )
    # 実行コンテキストのrun_configからローカルエポック数を取得します。なければデフォルト値1を使用します。
    local_epochs = int(context.run_config.get("local-epochs", 1))

    # FlowerClientをインスタンス化し、.to_client()メソッドでFlowerが扱える形式に変換して返します。
    return FlowerClient(
        net, trainloader, valloader, local_epochs, num_partitions
    ).to_client()


# ClientAppを作成します。Flowerはこのappオブジェクトを使ってクライアントアプリケーションを実行します。
app = ClientApp(
    client_fn=client_fn,  # クライアントを生成する関数としてclient_fnを指定します。
)
