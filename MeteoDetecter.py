

#!/ python3
# atomcam.py by Hase-kin ストリーミング（スレッド版） を=== USBカメラ用にに構成
# オプションで切り替えられるURLの値をUSBキャプチャーカメラを指定する番号"１”などにした
#　PlayeroneのCMOSカメラなどで流星検知に使用できる。


from pathlib import Path
import sys
import os
from datetime import datetime, timedelta, timezone
import time
import argparse
import numpy as np
import cv2
from imutils.video import FileVideoStream

# マルチスレッド関係
import threading
import queue


# 行毎に標準出力のバッファをflushする。
sys.stdout.reconfigure(line_buffering=True)


def composite(list_images):
    """画像リストの合成(単純スタッキング)
    Args:
      list_images: 画像データのリスト
    Returns:
      合成された画像
    """
    equal_fraction = 1.0 / (len(list_images))

    output = np.zeros_like(list_images[0])

    for img in list_images:
        output = output + img * equal_fraction

    output = output.astype(np.uint8)

    return output


def brightest(img_list):
    """比較明合成処理
    Args:
      img_list: 画像データのリスト
    Returns:
      比較明合成された画像
    """
    output = img_list[0]

    for img in img_list[1:]:
        output = np.where(output > img, output, img)

    return output


def diff(img_list, mask):
    """画像リストから差分画像のリストを作成する。
    Args:
      img_list: 画像データのリスト
      mask: マスク画像(2値画像)
    Returns:
      差分画像のリスト
    """
    diff_list = []
    for img1, img2 in zip(img_list[:-2], img_list[1:]):
        # img1 = cv2.bitwise_or(img1, mask)
        # img2 = cv2.bitwise_or(img2, mask)
        diff_list.append(cv2.subtract(img1, img2))

    return diff_list


def detect(img, min_length):
    """画像上の線状のパターンを流星として検出する。
    Args:
      img: 検出対象となる画像
      min_length: HoughLinesPで検出する最短長(ピクセル)
    Returns:
      検出結果
    """
    blur_size = (5, 5)
    blur = cv2.GaussianBlur(img, blur_size, 0)
    canny = cv2.Canny(blur, 100, 200, 3)

    # The Hough-transform algo:
    return cv2.HoughLinesP(canny, 1, np.pi/180, 25, minLineLength=min_length, maxLineGap=5)


class Meteor:
    def __init__(self, video_url, output=None, end_time="0600", minLineLength=30):
        self._running = False
        # video device url or movie file path
        self.capture = None
        self.source = None
        self.mask = None
        self.url = video_url

        self.connect()
        self.FPS = int(self.capture.get(cv2.CAP_PROP_FPS))
        print(self.FPS)
        self.HEIGHT = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.WIDTH = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(self.WIDTH)

        # 出力先ディレクトリ
        if output:
            output_dir = Path(output)
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = Path('.')
        self.output_dir = output_dir

        self.mp4 = False

        # 終了時刻を設定する。
        now = datetime.now()
        t = datetime.strptime(end_time, "%H%M")
        self.end_time = datetime(now.year, now.month, now.day, t.hour, t.minute)
        if now > self.end_time:
            self.end_time = self.end_time + timedelta(hours=24)

        print("# scheduled end_time = ", self.end_time)
        self.now = now

        self.min_length = minLineLength
        self.image_queue = queue.Queue(maxsize=200)

    def __del__(self):
        now = datetime.now()
        obs_time = "{:04}/{:02}/{:02} {:02}:{:02}:{:02}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second
        )
        print("# {} stop".format(obs_time))

        self.capture.release()
        cv2.destroyAllWindows()

    def connect(self):
        if self.capture:
            self.capture.release()
        self.capture = cv2.VideoCapture(self.url)
        if self.url == 1:  # = USB Cmaera
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600) # Neptune-C 3096x2078
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400) # Mars-C 1944x1096
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            self.capture.set(cv2.CAP_PROP_EXPOSURE, -6)
            self.capture.set(cv2.CAP_PROP_GAIN,180)
            self.capture.set(cv2.CAP_PROP_GAMMA,1)


    def stop(self):
        # thread を止める
        self._running = False


    def queue_streaming(self):
        """RTSP読み込みをthreadで行い、queueにデータを流し込む。
        """
        print("# threading version started.")
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._running = True
        while(True):
            try:
                ret, frame = self.capture.read()
                frame = cv2.flip(frame,0)  # 上下反転（PlayeroneCamera)
                if ret:
                    # self.image_queue.put_nowait(frame)
                    now = datetime.now()
                    self.image_queue.put((now, frame))
                    if self.mp4:
                        current_pos = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
                        if current_pos >= frame_count:
                            break
                else:
                    self.connect()
                    time.sleep(5)
                    continue

                if self._running is False:
                    break
            except Exception as e:
                print(type(e), file=sys.stderr)
                print(e, file=sys.stderr)
                continue

    def dequeue_streaming(self, exposure=1, no_window=False):
        """queueからデータを読み出し流星検知、描画を行う。
        """
        num_frames = int(self.FPS * exposure)

        while True:
            img_list = []
            for n in range(num_frames):
                (t, frame) = self.image_queue.get()
                key = chr(cv2.waitKey(1) & 0xFF)
                if key == 'q':
                    self._running = False
                    return

                if self.mp4 and self.image_queue.empty():
                    self._running = False
                    return

                # exposure time を超えたら終了
                if len(img_list) == 0:
                    t0 = t
                    img_list.append(frame)
                else:
                    dt = t - t0
                    if dt.seconds < exposure:
                        img_list.append(frame)
                    else:
                        break

            if len(img_list) > 2:
                self.composite_img = brightest(img_list)
                if not no_window:
                    show_img = cv2.resize(self.composite_img, (1050, 700))
                    cv2.imshow('Meteor_Detecter'.format(self.source), show_img)
                self.detect_meteor(img_list)

            # ストリーミングの場合、終了時刻を過ぎたなら終了。
            now = datetime.now()
            if not self.mp4 and now > self.end_time:
                print("# end of observation at ", now)
                self._running = False
                return


    def detect_meteor(self, img_list):
        """img_listで与えられた画像のリストから流星(移動天体)を検出する。
        """
        now = datetime.now()
        obs_time = "{:04}/{:02}/{:02} {:02}:{:02}:{:02}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)

        if len(img_list) > 2:
            # 差分間で比較明合成を取るために最低3フレームが必要。
            # 画像のコンポジット(単純スタック)
            diff_img = brightest(diff(img_list, self.mask))
            try:
                if now.hour != self.now.hour:
                    # 毎時空の様子を記録する。
                    filename = "sky-{:04}{:02}{:02}{:02}{:02}{:02}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
                    path_name = str(Path(self.output_dir, filename + ".jpg"))
                    cv2.imwrite(path_name, self.composite_img)
                    self.now = now

                detected = detect(diff_img, self.min_length)
                if detected is not None:
                    '''
                    for meteor_candidate in detected:
                        print('{} {} A possible meteor was detected.'.format(obs_time, meteor_candidate))
                    '''
                    print('{} A possible meteor was detected.'.format(obs_time))
                    filename = "{:04}{:02}{:02}{:02}{:02}{:02}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
                    path_name = str(Path(self.output_dir, filename + ".jpg"))
                    cv2.imwrite(path_name, self.composite_img)

                    # 検出した動画を保存する。
                    movie_file = str(Path(self.output_dir, "movie-" + filename + ".mp4"))
                    self.save_movie(img_list, movie_file)
            except Exception as e:
                print(e, file=sys.stderr)

    def save_movie(self, img_list, pathname):
        """
        画像リストから動画を作成する。
        Args:
          imt_list: 画像のリスト
          pathname: 出力ファイル名
        """
        size = (self.WIDTH, self.HEIGHT)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        video = cv2.VideoWriter(pathname, fourcc, self.FPS, size)
        for img in img_list:
            video.write(img)

        video.release()


def streaming_thread(args):
    """
    RTSPストリーミング、及び動画ファイルからの流星の検出
    """
    if args.url:
        atom = Meteor(args.url, args.output, args.to, args.min_length)
        if not atom.capture.isOpened():
            return


    now = datetime.now()
    obs_time = "{:04}/{:02}/{:02} {:02}:{:02}:{:02}".format(
        now.year, now.month, now.day, now.hour, now.minute, now.second
    )
    print("# {} start".format(obs_time))

    # スレッド版の流星検出
    t_in = threading.Thread(target=atom.queue_streaming)
    t_in.start()

    try:
        atom.dequeue_streaming(args.exposure, args.no_window)
    except KeyboardInterrupt:
        atom.stop()

    t_in.join()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    # ストリーミングモードのオプション ==== url = 1 USB:VideoCamera
    parser.add_argument('-u', '--url', default = 1, help='デフォルトはUSB接続のカメラ：もともとはRTSPのURL、または動画(MP4)ファイル')
    parser.add_argument('-n', '--no_window', action='store_true', help='画面非表示')

    # 共通オプション
    parser.add_argument('-e', '--exposure', type=int, default=1, help='露出時間(second)')
    parser.add_argument('-o', '--output', default=None, help='検出画像の出力先ディレクトリ名')
    parser.add_argument('-t', '--to', default="0600", help='終了時刻(JST) "hhmm" 形式(ex. 0600)')
    parser.add_argument('--min_length', type=int, default=30, help="minLineLength of HoghLinesP")
    parser.add_argument('-s', '--suppress-warning', action='store_true', help='suppress warning messages')

    # threadモード
    parser.add_argument('--thread', default=False, action='store_true', help='スレッド版')
    parser.add_argument('--help', action='help', help='show this help message and exit')

    args = parser.parse_args()

    # atomcam.py by Hase-kin ストリーミング（スレッド版） を=== USBカメラ用にに構成
    streaming_thread(args)

