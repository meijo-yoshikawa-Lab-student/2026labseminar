"""task.py"""

# collectionsモジュールからOrderedDictをインポートします。順序付き辞書をモデルの重み管理に使用します。
from collections import OrderedDict
from typing import Any

# NumPyライブラリをインポートします。数値計算、特に配列操作に使用します。
from datasets.arrow_dataset import Dataset
import numpy as np

# PyTorchライブラリをインポートします。ニューラルネットワークの構築やテンソル演算に使用します。
import torch

# PyTorchのニューラルネットワークモジュールをnnという別名でインポートします。
import torch.nn as nn

# PyTorchのニューラルネットワーク関数モジュールをFという別名でインポートします。活性化関数などに使用します。
import torch.nn.functional as F

# Flower DatasetsライブラリからFederatedDatasetをインポートします。フェデレーテッドデータセットを簡単に扱うために使用します。
from flwr_datasets import FederatedDataset

# Flower DatasetsのパーティショナーからIidPartitionerをインポートします。データをIID（独立同分布）で分割するために使用します。
from flwr_datasets.partitioner import IidPartitioner

# PyTorchのユーティリティからDataLoaderをインポートします。データセットをバッチ単位で効率的に読み込むために使用します。
from torch.utils.data import DataLoader

# torchvisionのtransformsからCompose, Normalize, ToTensorをインポートします。画像データの前処理に使用します。
from torchvision.transforms import Compose, Normalize, ToTensor


def get_default_device():
    """CUDA、Apple Metal、CPUの順に利用可能なデバイスを選びます。"""

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Net(nn.Module):
    """ニューラルネットワークモデルを定義するクラスです。nn.Moduleを継承します。"""

    def __init__(self):
        """モデルの初期化メソッドです。層（レイヤー）をここで定義します。"""

        # 親クラス（nn.Module）の初期化メソッドを呼び出します。
        super(Net, self).__init__()
        # 最初の畳み込み層を定義します。入力チャネル1、出力チャネル32、カーネルサイズ3x3、パディング1です。
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 最初のバッチ正規化層を定義します。32チャネルに対応します。
        self.bn1 = nn.BatchNorm2d(32)
        # 最初のプーリング層を定義します。2x2の最大値プーリングです。
        self.pool1 = nn.MaxPool2d(2, 2)
        # 最初のドロップアウト層を定義します。ドロップアウト率は25%です。
        self.dropout1 = nn.Dropout(0.25)
        # 2番目の畳み込み層を定義します。入力チャネル32、出力チャネル64、カーネルサイズ3x3、パディング1です。
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 2番目のバッチ正規化層を定義します。64チャネルに対応します。
        self.bn2 = nn.BatchNorm2d(64)
        # 2番目のプーリング層を定義します。2x2の最大値プーリングです。
        self.pool2 = nn.MaxPool2d(2, 2)
        # 2番目のドロップアウト層を定義します。ドロップアウト率は25%です。
        self.dropout2 = nn.Dropout(0.25)
        # 平坦化層を定義します。多次元のデータを1次元に変換します。
        self.flatten = nn.Flatten()
        # 最初の全結合層（線形層）を定義します。入力次元64*7*7、出力次元128です。
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 3番目のドロップアウト層を定義します。ドロップアウト率は50%です。
        self.dropout3 = nn.Dropout(0.5)
        # 2番目の全結合層を定義します。出力層であり、10クラス分類なので出力次元は10です。
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """順伝播を定義するメソッドです。データがモデルを通過する流れを記述します。"""

        # 入力xをconv1 -> bn1 -> relu -> pool1の順に処理します。
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # ドロップアウトを適用します。
        x = self.dropout1(x)
        # データをconv2 -> bn2 -> relu -> pool2の順に処理します。
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # ドロップアウトを適用します。
        x = self.dropout2(x)
        # テンソルを平坦化（1次元化）します。
        x = self.flatten(x)
        # データをfc1 -> reluの順に処理します。
        x = F.relu(self.fc1(x))
        # ドロップアウトを適用します。
        x = self.dropout3(x)
        # 最終的な出力層fc2を通過させます。
        x = self.fc2(x)
        # モデルの出力を返します。
        return x


# FederatedDatasetオブジェクトを格納するためのグローバル変数を初期化します。
fds = None


def load_data(
    partition_id: int,  # クライアントに対応するデータパーティションのID
    num_partitions: int,  # 全体のパーティション数
    random_seed: int = 42,  # データ分割に使用する乱数シード
):
    """データを読み込むための関数を定義します。"""

    # グローバル変数のfdsを使用することを宣言します。
    global fds
    # fdsがまだ読み込まれていない場合（初回呼び出し時）にのみ実行します。
    if fds is None:
        # IID（独立同分布）でデータを分割するパーティショナーを作成します。
        partitioner = IidPartitioner(num_partitions=num_partitions)
        # FederatedDatasetを使って"mnist"データセットを読み込み、訓練データを指定したパーティショナーで分割します。
        fds = FederatedDataset(
            dataset="mnist",  # データセット名
            partitioners={"train": partitioner},  # 訓練データを分割
        )

    # 指定されたpartition_idのデータパーティションを読み込みます。
    try:
        # "train"スプリットから指定のIDのパーティションをロードします。
        partition = fds.load_partition(partition_id, "train")
    # 不正なpartition_idが指定された場合のエラーハンドリングです。
    except IndexError as e:
        # エラーメッセージを出力します。
        print(
            f"Error loading partition {partition_id} for dataset 'mnist' (train split). "
            f"Ensure 'partition_id' (0-{num_partitions - 1}) is valid. Original error: {e}"
        )
        # エラーを再度送出します。
        raise
    # その他の予期せぬエラーが発生した場合のハンドリングです。
    except Exception as e:
        # エラーメッセージを出力します。
        print(
            f"An unexpected error occurred while loading partition {partition_id} for 'mnist' (train split): {e}"
        )
        # エラーを再度送出します。
        raise

    # パーティションを訓練データとテストデータに8:2の比率で分割します。
    partition_train_test = partition.train_test_split(test_size=0.2, seed=random_seed)

    # 画像に適用する一連の前処理を定義します。
    pytorch_transforms = Compose(
        [
            ToTensor(),  # PIL画像をPyTorchテンソルに変換します。
            Normalize(
                (0.1307,), (0.3081,)
            ),  # MNISTデータセットの平均と標準偏差で正規化します。
        ]
    )

    def apply_transforms(batch):
        """バッチデータに前処理を適用する関数を定義します。"""

        # バッチ内の各"image"に定義した前処理を適用します。
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        # 処理後のバッチを返します。
        return batch

    # データセット全体にtransform関数を適用します。
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    # 分割されたデータから訓練用データセットを取得します。
    train_dataset = partition_train_test["train"]

    # 訓練データセットのサンプル数がバッチサイズ以上の場合、最後の不完全なバッチを破棄します。
    drop_last = len(train_dataset) >= 32
    # 訓練データ用のDataLoaderを作成します。バッチサイズ32、データをシャッフルします。
    trainloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, drop_last=drop_last  # type: ignore[reportArgumentType]
    )
    # テストデータ用のDataLoaderを作成します。バッチサイズは32です。
    testloader = DataLoader(partition_train_test["test"], batch_size=32) # type: ignore[reportArgumentType]

    # 作成した訓練用とテスト用のDataLoaderを返します。
    return trainloader, testloader


def train(net, trainloader, epochs, device, learning_rate=0.001):
    """モデルを訓練する関数を定義します。"""

    # モデルを指定されたデバイス（CPUまたはGPU）に移動します。
    net.to(device)
    # Adamオプティマイザを初期化します。モデルのパラメータと学習率を渡します。
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # 損失関数としてクロスエントロピー損失を定義し、デバイスに移動します。
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # 各エポックの損失を格納するためのリストを初期化します。
    epoch_losses = []

    # モデルを訓練モードに設定します。
    net.train()
    # 指定されたエポック数だけループします。
    for epoch in range(epochs):
        # 現在のエポックの累積損失を初期化します。
        running_loss = 0.0
        # バッチ数を初期化します。
        num_batches = 0
        # trainloaderが空、またはデータセットが空の場合の警告処理です。
        if not trainloader or len(trainloader.dataset) == 0:
            # 最初のepochでのみ警告を表示します。
            if epoch == 0:
                # 警告メッセージを出力します。
                print(f"Warning: No training batches to process in epoch {epoch + 1}.")
            # このエポックの損失を0.0として追加します。
            epoch_losses.append(0.0)
            # 次のエポックに進みます。
            continue

        # 訓練データローダーからバッチ単位でデータを取得してループします。
        for batch_idx, batch_data in enumerate(trainloader):
            # 画像データを取得し、指定デバイスに移動します。
            images = batch_data["image"].to(device)
            # ラベルデータを取得し、指定デバイスに移動します。
            labels = batch_data["label"].to(device)

            # オプティマイザの勾配をリセットします。
            optimizer.zero_grad()
            # モデルに画像データを入力し、出力を得ます（順伝播）。
            outputs = net(images)
            # 出力と正解ラベルから損失を計算します。
            loss = criterion(outputs, labels)
            # 損失に基づいて勾配を計算します（逆伝播）。
            loss.backward()
            # 計算された勾配に基づいてモデルの重みを更新します。
            optimizer.step()

            # 損失を累積します。
            running_loss += loss.item()
            # バッチ数をインクリメントします。
            num_batches += 1

        # バッチが1つ以上処理された場合
        if num_batches > 0:
            # エポックの平均損失を計算します。
            epoch_loss = running_loss / num_batches
            # 計算したエポック損失をリストに追加します。
            epoch_losses.append(epoch_loss)
        # データセットは存在するがバッチが生成されなかった場合の処理
        elif len(trainloader.dataset) > 0:
            # エポック損失を0.0としてリストに追加します。
            epoch_losses.append(0.0)
            # 最初のepochでのみ警告を表示します。
            if epoch == 0:
                # 警告メッセージを出力します。
                print(
                    f"Warning: Training batches were not generated in epoch {epoch + 1} (num_batches=0). Dataset size: {len(trainloader.dataset)}"
                )
        # データセットも空の場合
        else:
            # エポック損失を0.0としてリストに追加します。
            epoch_losses.append(0.0)

    # 全エポックの平均損失を計算して返します。リストが空の場合は0.0を返します。
    return sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0


def test(net, testloader, device):
    """モデルの性能を評価する関数を定義します。"""

    # モデルを指定されたデバイス（CPUまたはGPU）に移動します。
    net.to(device)
    # 損失関数としてクロスエントロピー損失を定義します。
    criterion = torch.nn.CrossEntropyLoss()
    # 正解数、累積損失、合計サンプル数を初期化します。
    correct, loss_sum, total_samples = 0, 0.0, 0
    # モデルを評価モードに設定します。
    net.eval()
    # 勾配計算を無効にするコンテキストで実行します。
    with torch.no_grad():
        # testloaderが空、またはデータセットが空の場合の警告処理です。
        if not testloader or len(testloader.dataset) == 0:
            # 警告メッセージを出力します。
            print("Warning: Test data loader is empty.")
            # 損失0.0、正解率0.0を返します。
            return 0.0, 0.0

        # テストデータローダーからバッチ単位でデータを取得してループします。
        for batch in testloader:
            # バッチが空、または画像データが含まれていない場合はスキップします。
            if (
                not batch
                or "image" not in batch
                or batch["image"] is None
                or batch["image"].shape[0] == 0
            ):
                # 次のバッチに進みます。
                continue
            # 画像データを取得し、指定デバイスに移動します。
            images = batch["image"].to(device)
            # ラベルデータを取得し、指定デバイスに移動します。
            labels = batch["label"].to(device)
            # モデルに画像データを入力し、出力を得ます。
            outputs = net(images)
            # バッチの損失を計算し、累積損失に加算します。
            loss_sum += criterion(outputs, labels).item() * len(labels)
            # 合計サンプル数を更新します。
            total_samples += len(labels)
            # 最も確率の高いクラスを予測結果とし、正解数を加算します。
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    # 正解率を計算します。合計サンプル数が0の場合は0とします。
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    # 平均損失を計算します。合計サンプル数が0の場合は0とします。
    avg_loss = loss_sum / total_samples if total_samples > 0 else 0.0
    # 平均損失と正解率を返します。
    return avg_loss, accuracy


def get_weights(net):
    """モデルの重み（パラメータ）を取得する関数を定義します。"""

    # モデルの状態辞書から値（重みテンソル）を取り出し、NumPy配列に変換してリストとして返します。
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """モデルに重み（パラメータ）を設定する関数を定義します。"""

    # モデルの状態辞書のキーと、受け取ったパラメータリストをペアにします。
    params_dict = zip(net.state_dict().keys(), parameters)
    # パラメータの各値をPyTorchテンソルに変換し、順序付き辞書（state_dict）を構築します。
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    # 構築した状態辞書をモデルに読み込ませます。strict=Trueはキーの完全一致を要求します。
    net.load_state_dict(state_dict, strict=True)
