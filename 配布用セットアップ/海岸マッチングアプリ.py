import PySimpleGUI as sg
from geopy.distance import geodesic
from pycaret.regression import *
from PIL import Image, ImageTk
from fractions import Fraction as ft


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, ft('16/9') * min(pil_img.size), min(pil_img.size))

sg.theme('DarkGrey6')
font = ('meiryo', 12)
font2 = ('meiryo', 10)
box_font = ('meiryo', 8)
Goal_Place = range(1,101)


tab1_layout = [  [sg.Text('現在地の緯度:', font = font), sg.InputText(key='-NOWLATI-', size = (30,1)), sg.Text('現在地の経度', font = font), sg.InputText(key='-NOWLONG-', size = (30,1))],
            [sg.Button('マッチング', font = box_font, key = '-MACHING-')],
            [sg.Image(key="-IMAGE-")],
            [sg.Multiline(key = '-MULTILINE-', size = (100, 15), default_text = 'ここに海水浴場の情報が表示されます\n入力できないエリアです。', disabled = True)],
            [sg.Button('前の候補地', font = box_font, key = '-BACK-'), sg.Button('次の候補地', font = box_font, key = '-NEXT-')],
            [sg.Button('終了', font = box_font, key = '-END-')]  ]

tab2_layout = [  [sg.Text('～機械学習用の前処理のやり方～\n\n1.「海水浴場アンケートデータ.csv」を開き１列目の情報に従って情報を配置してください。'
                          '\n2.情報を配置したらファイルを保存し下の処理ボタンを押してください。'
                          '\n3.同じ階層にある処理層という名前のフォルダの中に処理済みファイルが保存されています。'
                          '\n4.juptter notebookで「マッチングアプリ学習モデル生成用.ipynb」を開き作業に移ってください。', font = font2)],
                [sg.Button('処理開始', font = box_font, key = '-PREPARATION-')],
                [sg.Button('終了', font = box_font, key = '-END2-')]  ]


layout = [  [sg.TabGroup([[sg.Tab('マッチングアプリ', tab1_layout, font = box_font), sg.Tab('機械学習用下処理', tab2_layout, font = box_font)]])]  ]
window = sg.Window('海水浴場マッチングアプリ', layout, size = (800,830))

#各データを読み込み
base_data = pd.read_csv('各海水浴場パラメータ.csv', index_col=0, encoding = 'utf-8', engine="python")
prep_data = pd.read_csv('海水浴場アンケートデータ.csv', index_col=0, encoding = 'utf-8', engine="python")

#PANDASのデータフレームの表示設定　※実際にはデータフレームを見ないのでテスト用の設定である事
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 30)
pd.set_option('display.unicode.east_asian_width', True)

#各海水浴場のパラメータデータフレームを加工
edit_data = base_data.drop(base_data.columns[[3, 4, 5, 6, 8]], axis=1)

#各変数
idx = len(edit_data.index)
idx_prep = len(prep_data.index)
size = (622, 350)
number = 0
count = 0


while True:
    event, values = window.read()

    if event in (sg.WIN_CLOSED, '-END-', '-END2-'):
        break

    elif event == '-PREPARATION-':
        for a in range(0, idx_prep):
            b = prep_data.iat[a, 6]  # 海水浴場の緯度
            c = prep_data.iat[a, 7]  # 海水浴場の経度
            d = prep_data.iat[a, 8]  # 現在位置の緯度
            e = prep_data.iat[a, 9]  # 現在位置の経度
            location = [float(b), float(c)]
            now_place = [d, e]
            prep_data.iat[a, 10] = geodesic(now_place, location).km
        learning_data = prep_data.drop(prep_data.columns[[6, 7, 8, 9]], axis=1)
        learning_data = learning_data.sample(frac =1)
        learning_data.to_excel('/マッチングアプリ/処理層/学習モデル生成用.xlsx')
        sg.popup_ok('学習モデル作成に使うアンケートデータの生成が完了しました。\n 次の作業に進んでください。')

    elif event == '-MACHING-':
        now_lati = values['-NOWLATI-']
        now_long = values['-NOWLONG-']
        count += 1

        if len(values['-NOWLATI-']) == 0 or len(values['-NOWLONG-']) == 0:
            sg.popup_error('現在地の緯度と経度を入力してください')  # エラーボタンを表示

        elif count == 2:
            pass

        elif count == 1:

            #学習モデルに入れるのに現在地の緯度経度を入力し、距離を計測してくれる部分
            for x in range(0, idx):
                edit_data.iat[x, 8] = float(now_lati)
                edit_data.iat[x, 9] = float(now_long)

            for a in range(0, idx):
                b = edit_data.iat[a, 6]  # 海水浴場の緯度
                c = edit_data.iat[a, 7]  # 海水浴場の経度
                d = edit_data.iat[a, 8]  # 現在位置の緯度
                e = edit_data.iat[a, 9]  # 現在位置の経度
                location = [float(b), float(c)]
                now_place = [d, e]
                edit_data.iat[a, 10] = geodesic(now_place, location).km
            edit_data = edit_data.drop(edit_data.columns[[6, 7, 8, 9]], axis=1)

            # 学習モデルに数字を入れ評価予測値を出す
            newmodel = load_model('matching_app_learning')
            new_prediction = predict_model(newmodel, data = edit_data)
            sort_data = new_prediction.sort_values(by = 'Label', ascending=False)

            # 海水浴場名リストを生成
            beach_name_index = sort_data.index.values.tolist()
            best_beach = beach_name_index[number]

            #海水浴場情報の各パラメータ
            width = sort_data.iat[number, 1]
            v_width = sort_data.iat[number,2]
            users = sort_data.iat[number,3]
            water_quality = sort_data.iat[number, 4]
            distance = sort_data.iat[number, 6]
            evaluation = sort_data.iat[number, 7]

            #海水浴場の現地写真を読み込み画面に表示させる
            im = Image.open(f'写真/{best_beach}.png')
            im_resize = crop_max_square(im)
            im_finish = im_resize.resize(size, resample = Image.LANCZOS)
            image = ImageTk.PhotoImage(image=im_finish)

            window['-IMAGE-'].update(data=image)
            window['-MULTILINE-'].update(f'あなたにおすすめな海水浴場は{best_beach}です。\n\n情報\n{number + 1}番目の候補\n\n横幅:{width}m\n縦幅:{v_width}m\n一日当たりの利用者数:{users}人\n水質:{water_quality}\n予測評価値:{round(evaluation, 2)}/10')


    elif event == '-BACK-':
        if number == 0 or count == 0:
            pass

        else:
            number = number - 1

            a_beach_name = beach_name_index[number]
            width = sort_data.iat[number, 1]
            v_width = sort_data.iat[number, 2]
            users = sort_data.iat[number, 3]
            water_quality = sort_data.iat[number, 4]
            distance = sort_data.iat[number, 6]
            evaluation = sort_data.iat[number, 7]

            im = Image.open(f'写真/{a_beach_name}.png')
            im_resize = crop_max_square(im)
            im_finish = im_resize.resize(size, resample=Image.LANCZOS)
            image = ImageTk.PhotoImage(image=im_finish)

            window['-IMAGE-'].update(data=image)
            window['-MULTILINE-'].update(f'あなたにおすすめな海水浴場は{a_beach_name}です。\n\n情報\n{number + 1}番目の候補\n\n横幅:{width}m\n縦幅:{v_width}m\n一日当たりの利用者数:{users}人\n水質:{water_quality}\n予測評価:{round(evaluation, 2)}/10')


    elif event == '-NEXT-':
        if number == idx - 1 or count == 0:
            pass

        else:
            number = number + 1

            b_beach_name = beach_name_index[number]
            width = sort_data.iat[number, 1]
            v_width = sort_data.iat[number, 2]
            users = sort_data.iat[number, 3]
            water_quality = sort_data.iat[number, 4]
            distance = sort_data.iat[number, 6]
            evaluation = sort_data.iat[number, 7]

            im = Image.open(f'写真/{b_beach_name}.png')
            im_resize = crop_max_square(im)
            im_finish = im_resize.resize(size, resample=Image.LANCZOS)
            image = ImageTk.PhotoImage(image=im_finish)

            window['-IMAGE-'].update(data=image)
            window['-MULTILINE-'].update(f'あなたにおすすめな海水浴場は{b_beach_name}です。\n\n情報\n{number + 1}番目の候補\n\n横幅:{width}m\n縦幅:{v_width}m\n一日当たりの利用者数:{users}人\n水質:{water_quality}\n予測評価:{round(evaluation, 2)}/10')

window.close()



