#!/usr/bin/env python3
"""Build the six-page A4 submission from existing measured evidence.

python3 scripts/build_report.py
python3 scripts/build_report.py --html-only
"""
from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

from submission_assets import ROOT, OUT, ROUTES, accuracy, build_assets

STEM = '説明資料_PAI最終課題_142245'


def esc(value):
    return html.escape(str(value), quote=True)


def figure(name, caption, css=''):
    data = base64.b64encode((OUT / name).read_bytes()).decode('ascii')
    return (f'<figure class="{css}"><img src="data:image/png;base64,{data}" alt="{esc(caption)}">'
            f'<figcaption>{esc(caption)}</figcaption></figure>')


def table(headers, rows):
    return '<table><thead><tr>' + ''.join(f'<th>{esc(x)}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{esc(x)}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'


def page(number, title, body):
    return f'<section class="page"><p class="kicker">PHYSICAL AI 応用1 / 最終課題 / 142245</p><h1><span>{number:02d}</span> {title}</h1>{body}</section>'


def render_report(c, t):
    cnn, bright = c['test']['cnn'], c['test']['brightness']
    threshold = c['review_policy']['threshold']
    cycles = t['cycles']
    n = len(cycles)
    successes = sum(r['transfer_success'] is True for r in cycles)
    accepted = [r for r in cycles if r['destination'] != 'review']
    accepted_correct = sum(r['correct_route'] is True for r in accepted)
    review = n - len(accepted)
    splits = c['dataset']['split_summary']
    comparison = []
    for condition, label in [(None, '全体'), ('nominal', '通常照明'), ('dark', '暗照明')]:
        items = [c['test'][m] if condition is None else c['test'][m]['lighting'][condition] for m in ('cnn', 'brightness')]
        comparison.append((label, accuracy(items[0]), accuracy(items[1])))
    routes = []
    for route, (name, _) in ROUTES.items():
        rows = [r for r in cycles if r['destination'] == route]
        routes.append((name, len(rows), sum(r['transfer_success'] is True for r in rows),
                       '判断保留' if route == 'review' else '良否を確定'))
    trial_rows = []
    for r in cycles[:3]:
        d = r['decision']
        trial_rows.append((r['trial'], f'{d["defect_score"]:.6f}', f'{d["confidence"]:.6f}', ROUTES[r['destination']][0]))
    parts = []
    parts.append(page(1, 'AI外観検査・仕分けセル', f'''
<p class="lead">画像で汚れを見分け、判断を保留できる<br>ロボット仕分けセルを作る。</p>
<h2>概要</h2>
<p>製造業の外観検査を題材に、部品の撮影からAIによる良否判定、ロボットによる仕分け、配置確認・記録までをつないだ。シミュレータ上のロボットアームが、直方体部品の上面にある模擬汚れを画像から判定し、良品・不良品・要確認のトレイへ運ぶ。AIが十分な確信を持てないときは良否を決めず、要確認として分離する。</p>
{figure('cell.png', '図1　UR10eとRobotiq 2F-85による搬送。緑：良品、赤：不良品、黄：要確認。見やすさのため、この全景のみ表示の明るさを補正。', 'hero')}
<h2>背景と目的</h2>
<p>外観検査では、見た目の判定と、その結果に応じた物理的な仕分けを一続きにする必要がある。本課題では「画像を分類できる」だけでなく、「判定どおりに運び、手を離した後の状態まで確かめる」ことを目指した。また、判定を保留する仕組みが、照明変化による誤りをどこまで抑えられるかを調べた。</p>
<div class="callout"><b>確認できたこと：</b>独立テストのCNN正解率は{accuracy(cnn)}。選定した固定条件での搬送は{successes}/{n}件成功。分類性能と搬送性能は、それぞれ異なる実験で評価した。</div>
<p class="small">対象はシミュレーション内の固定位置・固定姿勢の部品。実機・連続供給・未知姿勢への対応は評価していない。</p>
'''))
    parts.append(page(2, 'システム構成とAIの役割', f'''
<h2>撮影から確認までの流れ</h2>
{figure('pipeline.png', '図2　画像から行先を決め、搬送後の状態を確認して記録する。')}
{table(('段階', '処理と役割'), [
('カメラ', '固定カメラで192×192のRGB画像を撮影。部品上面の模擬汚れを観察する。'),
('CNN', '画像を64×64に縮小し、小型の畳み込みニューラルネットワークで不良スコアを出す。'),
('判定', '確信度が基準未満なら要確認。基準以上なら、不良スコア0.5を境に良品・不良品を決める。'),
('搬送', 'UR10eと2指グリッパーが、接近・把持・持上げ・搬送・開放・退避を行う。'),
('確認・ログ', '部品全体の位置と速度、指の開放を確認し、画像・スコア・行先・成否を保存する。')])}
<h2>AIが担当する部分</h2>
<p>CNNは画像中の模擬汚れを判定する。入力はRGB画素だけで、部品名・正解ラベル・照明設定・目的トレイを渡さない。モデルは学習用画像から学習し、推論結果を搬送先の切替に使う。ロボットの経路制御には、既知の供給位置とトレイ位置を用いる。</p>
<h2>使用した環境・ツール</h2>
{table(('用途', '使用したもの'), [
('シミュレーション', c['environment']['simulator'] + ' / UR10e + Robotiq 2F-85'),
('学習・推論', 'Python / PyTorch / NumPy / Pillow。CPUでローカル学習。'),
('ロボット制御', '手先の直線経路と逆運動学（目標手先位置から関節角を求める計算）。'),
('実行環境', c['environment']['hardware'] + ' / Linux ' + c['environment']['architecture']),
('提出資料の生成', '既存の画像・ログをPillow / ffmpeg / WeasyPrintで編集。')])}
<p class="small">外部AI APIは使用していない。シミュレータはロボット・接触・描画を担当し、CNNは画像からの判定を担当する。</p>
'''))
    parts.append(page(3, '工夫した点', f'''
<h2>1. 画像だけの判定と、判断を保留する経路</h2>
<p>判定と搬送計測を分け、シミュレータが知っている正解をAIへ渡さない。確信度は、不良スコアをpとしてmax(p, 1−p)で定義した。検証用データ（val）で決めた基準を下回った場合、要確認トレイに運んで良否の確定を保留する。</p>
{figure('inspection_inputs.png', '図3　動画の3試行で使った検査画像を、CNN入力と同じ64×64へ縮小して拡大表示。画素の明るさ・色は補正していない。')}
{table(('動画の試行', '不良スコア p', '確信度', '行先'), trial_rows)}
<div class="callout"><b>試行3が要確認になる理由：</b>確信度{cycles[2]['decision']['confidence']:.6f}が、valで固定した基準{threshold:.6f}を下回るため。pが0.5以上でも、不良品と確定する前に保留の条件を確認する。表示値は小数点以下6桁に丸めている。</div>
<h2>2. 条件単位で分け、同じ画像で比較する</h2>
<p>色・照明・汚れ位置などの生成条件ごとに良品と不良品を対にし、同じ条件を学習・検証・テストにまたがらせない。CNNと単純な輝度方式を同じテスト画像で比べ、AIによる改善と苦手な条件を確認した。</p>
<h2>3. 「動作終了」と「搬送成功」を分ける</h2>
<p>手先が目標に着いただけでは成功としない。持上げ、部品全体のトレイ内配置、開放・退避後の静止を確認する。失敗理由と検査画像を試行ごとに保存し、判定と動作を後から追えるようにした。</p>
'''))
    parts.append(page(4, '実験結果：画像分類', f'''
<h2>実験方法</h2>
<p>生成画像を学習用train、設定を決める検証用val、最後に性能を測るtestに分けた。モデル・要確認基準はvalで固定し、testを見て変更していない。各条件の良品・不良品は同じ分割に置く。</p>
{table(('分割', '画像数', '条件グループ数', '用途'), [(label, splits[key]['images'], splits[key]['groups'], purpose) for key, label, purpose in [('train', '学習用', 'モデルの重みを学習'), ('val', '検証用', 'モデルと要確認基準を選択'), ('test', 'テスト用', '固定後の性能を測定')]])}
<p>比較の輝度方式は、部品上面の中央領域で暗い画素の明るさを求め、しきい値で良否を判定する。テストには、通常照明に加え、学習・検証時より暗い照明条件を含めた。</p>
{figure('classification.png', '図4　全画像を良品か不良品に二分したときの正解率。要確認として除外した後の値ではない。')}
{table(('テスト条件', 'CNN', '輝度方式'), comparison)}
<h2>要確認を使った場合</h2>
{table(('テスト全体', 'CNN', '輝度方式'), [
('要確認に保留した画像', f'{cnn["review_count"]}/{cnn["samples"]}', f'{bright["review_count"]}/{bright["samples"]}'),
('良否を確定した画像', f'{cnn["samples"]-cnn["review_count"]}/{cnn["samples"]}', f'{bright["samples"]-bright["review_count"]}/{bright["samples"]}'),
('確定した画像中の誤り', f'{cnn["routing_confusion"]["matrix"][0][1] + cnn["routing_confusion"]["matrix"][1][0]}/{cnn["samples"]-cnn["review_count"]}（{cnn["accepted_error"]*100:.1f}%）', '未定義（確定した画像なし）')])}
<p class="small">輝度方式はvalで所定の誤り率基準を満たす判定範囲を選べず、全件を保留した。保留を増やせば自動判定の件数も減るため、正解率と保留件数を併記した。</p>
'''))
    parts.append(page(5, '実験結果：搬送と接続', f'''
<div class="metric"><strong>{successes}/{n}</strong><span>画像判定から搬送・開放後静止まで完了</span></div>
<h2>この数値が表す条件</h2>
<p>valから良品・不良品・要確認の経路を確認できる外観条件を選び、固定した3条件を繰り返した。行先は毎回再撮影した画像への推論で決める。各試行はロボット関節・部品位置・姿勢を初期状態に戻して開始する。</p>
{table(('画像判定の行先', '試行数', '搬送成功', '扱い'), routes)}
<p>良否を確定した{len(accepted)}件は{accepted_correct}/{len(accepted)}件で正解の行先と一致し、{review}件は要確認へ保留した。要確認への搬送は処理の完了であり、良否分類の正解数には加えない。</p>
<div class="callout">{successes}/{n}は、選定した固定条件での物理搬送・AI接続の結果。未知条件の分類正解率、連続供給機の稼働率、トレイへの20部品蓄積、実機の成功率を示すものではない。</div>
<h2>厳密な搬送成功の確認</h2>
{table(('確認する状態', '成功条件'), [
('持上げ', '部品が供給位置より0.06 mを超えて高くなる。'),
('全体の配置', '回転後も部品全体を囲む直方体が、指定トレイ内部に収まる。中心点だけでは判定しない。'),
('床との距離', '部品下面がトレイ床から0.012 m以内にある。'),
('開放・退避', 'グリッパーが開き、部品から離れる。'),
('静止', '並進速度0.015 m/s未満、角速度0.15 rad/s未満が連続60ステップ続く（200 Hzで0.30秒）。')])}
<h2>映像とログの対応</h2>
<p>動画は、この20試行の最初の3試行を収録したもの。把持から退避までを既存フレームで示し、各試行の検査画像・不良スコア・行先を同じ試行のログから表示する。全景の明るさ補正は表示だけに適用し、AI入力の画素は変更していない。</p>
<p class="small">実測根拠：docs/transport_metrics.json と artifacts/evaluation20/metrics.json。両者の一致を生成時に検査する。物理搬送だけの調整試行や、旧構成の失敗試行をこの20件に混ぜていない。</p>
'''))
    dark = cnn['lighting']['dark']
    bad_good = dark['confusion']['matrix'][0][1]
    passed_bad = dark['routing_confusion']['matrix'][0][1]
    parts.append(page(6, '考察・限界・今後', f'''
<h2>暗照明で分かった限界</h2>
<p>暗照明では良品{bad_good}枚すべてを不良品と誤認し、そのうち{passed_bad}枚は要確認にも入らず判定された。通常照明では識別できても、未経験の照明で高い確信度の誤りが起こる。確信度は正答確率の保証ではなく、今回の要確認基準だけでは照明変化を十分に検出できなかった。</p>
<p>原因として、モデルが汚れの形状だけでなく表面の暗さにも依存した可能性がある。ただし、寄与の切り分けは未実施である。一部の良品画像には描画履歴に由来する薄い矩形の残像もあり、実物の清浄な表面を完全に再現したデータではない。</p>
<h2>改善案と実機化に必要なこと</h2>
{table(('課題', '今後の改善・検証'), [
('照明変化', '暗照明を含む多様な条件で学習・検証し、別の未使用テストで再評価する。照明異常を別に検出して停止・再撮影する方法も検討する。'),
('合成画像の残像', '描画状態の初期化を見直し、良品画像を点検する。残像の有無を分けた比較で影響を調べる。'),
('供給位置・姿勢の固定', '部品の位置・姿勢推定と供給機構を追加し、姿勢や寸法の違いを含む把持試験を行う。'),
('実機での未検証', '実カメラ画像を収集し、カメラとロボットの座標対応、把持力・滑り、接触、非常停止・安全対策を検証する。')])}
<h2>参考・流用元と今回の追加部分</h2>
<p class="small">NVIDIA Isaac SimのUR10eサンプルに由来するロボット設定・搬送制御を利用・変更した。出典：<a href="https://github.com/isaac-sim/IsaacSim/tree/main/source/standalone_examples/api/isaacsim.robot.manipulators/ur10e">NVIDIA Isaac Sim UR10eサンプル</a>。対象コードのライセンスはApache-2.0。著作権・ライセンス表示を保持し、詳細な流用範囲と変更点をTHIRD_PARTY_NOTICES.md、ライセンス本文をLICENSES/Apache-2.0.txtに記載した。</p>
<p class="small">今回追加したのは、検査シーン、画像生成と条件分離、小型CNN、輝度比較、要確認への振り分け、厳密な搬送計測、検査ログ・録画。Isaac Sim本体、ロボットの形状・メッシュなどのアセットは同梱せず実行時に参照する。これらのアセットの利用条件は、コードのApache-2.0とは別にNVIDIAの配布条件に従う。</p>
<p class="small">測定根拠：docs/classification_metrics.json、docs/transport_metrics.json。実験の詳細：docs/CLASSIFICATION.md、docs/RESULTS.md。測定日：{esc(c['measured_date'])}。</p>
<div class="links"><b>提出先リンク</b><br>GitHub URL：（提出時に記入）<br>動画 URL：（提出時に記入）<br>動画タイトル：PAI最終課題_142245</div>
'''))
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'artifacts/report')
    parser.add_argument('--html-only', action='store_true')
    args = parser.parse_args()
    c, t = build_assets()
    css = (ROOT / 'docs/report.css').read_text()
    document = ('<!doctype html><html lang="ja"><head><meta charset="utf-8">'
                f'<title>{STEM}</title><style>{css}</style></head><body>'
                + render_report(c, t) + '</body></html>')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f'{STEM}.html').write_text(document, encoding='utf-8')
    if not args.html_only:
        from weasyprint import HTML
        rendered = HTML(string=document).render()
        if len(rendered.pages) != 6:
            raise ValueError(f'Expected 6 pages; found {len(rendered.pages)}. Check layout.')
        rendered.write_pdf(args.out_dir / f'{STEM}.pdf')
    print('Submission report generated: ' + STEM)


if __name__ == '__main__':
    main()
