#!/usr/bin/env python3
"""Build an anonymous four-page A4 report from measured local artifacts.

python3 scripts/build_report.py
python3 scripts/build_report.py --html-only

Missing metrics stay unmeasured. Requested trial counts never become measured
outcomes. No SimulationApp, remote API, font download or asset copy is used.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import re
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATH = re.compile(r'(?:/home/|/Users/|[A-Za-z]:\\Users\\)[^\s<>\]`"\')]+')


def clean_text(value):
    return PRIVATE_PATH.sub('[個人パス省略]', str(value))


def esc(value):
    return html.escape(clean_text(value), quote=True)


def load_optional(path):
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path.name}: JSON object required')
    return value


def percent(value, missing='未測定'):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f'{100 * value:.1f}%'
    return missing


def number(value, digits=1):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return str(value) if isinstance(value, int) else f'{value:.{digits}f}'
    return '未測定'


def table(headers, rows, class_name=''):
    head = ''.join(f'<th>{esc(value)}</th>' for value in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{esc(value)}</td>' for value in row) + '</tr>' for row in rows)
    return f'<table class="{esc(class_name)}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def source_summary(path, limit=260):
    """A short prose excerpt; tables, code, figures and links are not imported."""
    if not path.is_file():
        return '資料は未作成。実測結果と残作業を確定して再ビルドする。'
    paragraphs, current, in_code = [], [], False
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('```'):
            in_code = not in_code
            continue
        if in_code or line.startswith(('#', '|', '![', '- ', '* ')):
            if current:
                paragraphs.append(' '.join(current))
                current = []
            continue
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(' '.join(current))
            current = []
    if current:
        paragraphs.append(' '.join(current))
    if not paragraphs:
        return '本文の要約は未記載。参照元資料で詳細を確認する。'
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', paragraphs[0])
    text = re.sub(r'[*`_]', '', text)
    text = clean_text(text)
    return text if len(text) <= limit else text[:limit] + '…'


def source_tasks(path):
    if not path.is_file():
        return []
    active, items = False, []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('#'):
            active = any(word in line for word in ('残作業', '手動', '未完', '今後', '提出前'))
        elif active and re.match(r'^\s*[-*]\s', line):
            text = re.sub(r'^\s*[-*]\s(?:\[[ xX]\]\s)?', '', line)
            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
            items.append(clean_text(text.replace('`', ''))[:110])
            if len(items) == 4:
                break
    return items


def figure(path, caption, class_name=''):
    if not path or not path.is_file():
        return '<div class="missing-figure">図は未配置。最終画像を用意して再ビルドする。</div>'
    # Pixels are embedded, so the HTML/PDF needs no local absolute image URL.
    data = base64.b64encode(path.read_bytes()).decode('ascii')
    return (f'<figure class="{esc(class_name)}"><img src="data:image/png;base64,{data}" alt="{esc(caption)}">'
            f'<figcaption>{esc(caption)}</figcaption></figure>')


def measured_transfers(payload):
    if payload is None:
        return []
    rows = payload.get('cycles', payload.get('trials', []))
    if not isinstance(rows, list):
        raise ValueError('Transfer metrics require a cycles/trials list')
    return [row for row in rows if isinstance(row, dict) and isinstance(row.get('success'), bool)]


def transfer_summary(payload):
    rows = measured_transfers(payload)
    if not rows:
        return {'rows': [], 'count': 0, 'success': None, 'transfer': None, 'review': None,
                'success_rate': None, 'mean_wall': None, 'mean_sim': None, 'destinations': [], 'failures': []}
    destinations, reasons = [], Counter()
    for name in ('good', 'defect', 'review'):
        selected = [row for row in rows if row.get('destination') == name]
        passed = sum(row.get('transfer_success') is True for row in selected)
        destinations.append((name, len(selected), passed, f'{passed}/{len(selected)}' if selected else '未測定'))
    for row in rows:
        if not row['success']:
            reason = str(row.get('failure_reason') or row.get('reason') or '理由未記録')
            reasons[clean_text(reason)] += 1
    durations = {}
    for source, target in (('cycle_time', 'mean_wall'), ('simulation_seconds', 'mean_sim')):
        values = [row[source] for row in rows if isinstance(row.get(source), (int, float))
                  and not isinstance(row.get(source), bool) and math.isfinite(row[source]) and row[source] >= 0]
        durations[target] = mean(values) if values else None
    success = sum(row['success'] for row in rows)
    return {'rows': rows, 'count': len(rows), 'success': success,
            'transfer': sum(row.get('transfer_success') is True for row in rows),
            'review': sum(row.get('destination') == 'review' for row in rows),
            'success_rate': success / len(rows), 'destinations': destinations,
            'failures': reasons.most_common(4), **durations}


def classification_tables(metrics):
    if metrics is None:
        return '<p class="notice">画像分類の実測JSONが存在しないため未測定。</p>'
    rows = []
    for split, label in (('validation', 'val'), ('test', 'test')):
        for method, title in (('cnn', 'CNN'), ('brightness', '明度しきい値')):
            item = metrics.get(split, {}).get(method, {})
            rows.append((label, title, number(item.get('samples')), percent(item.get('accuracy')),
                         number(item.get('review_count')), percent(item.get('coverage')),
                         percent(item.get('accepted_error'), '未定義')))
    return table(('分割', '手法', '枚数', '二値正解率', '要確認', '採用率', '採用誤り率'), rows)


def confusion_table(metrics):
    if metrics is None:
        return '<p>混同行列は未測定。</p>'
    rows = []
    for method, title in (('cnn', 'CNN'), ('brightness', '明度しきい値')):
        matrix = metrics.get('test', {}).get(method, {}).get('confusion', {}).get('matrix')
        if not isinstance(matrix, list) or len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            continue
        rows.append((title, matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]))
    return table(('test手法', '良→良', '良→不良', '不良→良', '不良→不良'), rows)


def render_report(classification, transfers, project_root):
    observed = transfer_summary(transfers)
    dataset = classification.get('dataset', {}) if classification else {}
    training = classification.get('training', {}) if classification else {}
    environment = classification.get('environment', {}) if classification else {}
    split_rows = []
    for split in ('train', 'val', 'test'):
        item = dataset.get('split_summary', {}).get(split, {})
        split_rows.append((split, number(item.get('groups')), number(item.get('images'))))
    measured_date = classification.get('measured_date', '未測定') if classification else '未測定'
    status = transfers.get('status', '不明') if transfers else '未測定'
    requested = transfers.get('config', {}).get('trials_requested') if transfers else None
    state_note = f'記録状態: {clean_text(status)} / 実測済み {observed["count"]}試行。'
    if isinstance(requested, int):
        state_note += f' 要求件数（設定）は{requested}件であり、実測済み件数と区別する。'
    sample_image = project_root / 'review/inspection_samples.png'
    chart_image = project_root / 'review/classification_comparison.png'
    overview = next((project_root / name for name in ('review/cell_overview.png', 'review/sorter_overview.png',
                                                    'review/demo_overview.png') if (project_root / name).is_file()), None)
    model = f'{training.get("architecture", "小型CNN")} / {number(training.get("trainable_parameters"))} parameters'
    learning = f'CPU学習 {number(training.get("elapsed_seconds"))}秒 / 採用epoch {number(training.get("best_epoch"))}'
    cnn = classification.get('test', {}).get('cnn', {}) if classification else {}
    dark = cnn.get('lighting', {}).get('dark', {})
    dark_matrix = dark.get('confusion', {}).get('matrix')
    dark_note = '暗照明条件での分類評価は未測定。'
    if dark_matrix and len(dark_matrix) == 2:
        failed, total = dark_matrix[0][1], sum(dark_matrix[0])
        dark_note = (f'暗照明では良品{total}枚中{failed}枚を不良品と誤認した。'
                     f'暗照明全体の二値正解率は{percent(dark.get("accuracy"))}、'
                     f'採用分の誤り率は{percent(dark.get("accepted_error"), "未定義")}。'
                     'valで決めた確信度だけでは、この照明変化を十分検出できなかった。')
    if observed['count']:
        transfer_rows = [
            ('実測済み独立試行', observed['count']),
            ('物理搬送成功', f'{observed["transfer"]}/{observed["count"]}'),
            ('セル処理成功（保留含む）', f'{observed["success"]}/{observed["count"]} ({percent(observed["success_rate"])})'),
            ('要確認へ搬送した試行', observed['review']),
            ('平均の実処理 / シミュレーション秒', f'{number(observed["mean_wall"])} / {number(observed["mean_sim"])}'),
        ]
    else:
        transfer_rows = [('物理搬送・セル処理成功率', '未測定（実測済みの真偽値レコードなし）')]
    failure_html = ('<ul>' + ''.join(f'<li>{esc(reason)}: {count}件</li>' for reason, count in observed['failures']) + '</ul>'
                    if observed['failures'] else ('<p>記録済み試行内のセル処理失敗は0件。</p>'
                                                 if observed['count'] else '<p>失敗例の集計は未測定。</p>'))
    tasks = source_tasks(project_root / 'docs/RESULTS.md') or [
        '最終の搬送集計と短い試し動画を確認し、未達項目と失敗例を確定する。',
        '暗照明の誤分類を制約として説明し、実機への汎化は未評価とする。',
        '1〜3分の動画、public GitHub、PDFを本人が確認して提出する。',
    ]
    result_excerpt = (source_summary(project_root / 'docs/RESULTS.md') if observed['count']
                      else '搬送実測JSONが未存在、または実測済み試行がないため、搬送結果は未測定。')
    class_excerpt = (source_summary(project_root / 'docs/CLASSIFICATION.md') if classification
                     else '分類実測JSONが未存在のため、分類結果は未測定。')
    pages = [f'''
<section class="page">
  <p class="kicker">Physical AI 応用1講座 / 匿名の提出用下書き</p>
  <h1>AI外観検査・仕分けセル</h1>
  <p class="lead">画像だけで良否を判断し、ロボットが3トレイへ運び分ける。</p>
  <div class="pipeline">固定カメラ RGB → 小型CNN → 良品 / 不良品 / 要確認 → UR10e搬送 → 静止確認・検査ログ</div>
  <h2>目的と実装範囲</h2>
  <p>同形状の直方体の上面に模擬汚れを付け、RGB画像から不良スコアを推定する。判定に正解ラベル・物体名・物理状態を渡さない。valで固定した低確信度範囲は要確認へ送り、結果を画像・CSV・JSONで追跡する。</p>
  <div class="columns">
    <div>{table(('構成', '内容'), [
      ('検証環境', environment.get('simulator', 'Isaac Sim 6.0.0-rc.22') + ' / aarch64 / GB10'),
      ('ロボット', 'UR10e + native Robotiq 2F-85 variant'),
      ('分類器', model), ('学習', learning),
      ('制御', '200 Hz / Cartesian直線経路 / 画像判定で行先を切替'),
      ('課題の範囲', '固定位置・姿勢で1個ずつ。供給装置・実機は対象外'),
    ])}
    <h2>成果の読み方</h2><p>分類精度、物理搬送、判定から搬送までのセル処理を分ける。予定件数や静的FK一致を、成功した実測件数として扱わない。</p>
    <p class="small">画像分類の測定日: {esc(measured_date)}<br>搬送ログ: {esc(status)}</p></div>
    <div>{figure(sample_image, '固定カメラ画像。良品・模擬汚れありを同じ条件で対に生成。', 'samples')}</div>
  </div>
</section>''', f'''
<section class="page">
  <p class="kicker">1 / 画像分類の評価</p><h1>独立testと明度方式の比較</h1>
  <p>生成条件グループ単位でtrain / val / testを分離する。同条件の良品・不良品を同じ分割に保ち、testはモデルと要確認帯の確定後に評価する。</p>
  {table(('分割', '条件グループ', '画像枚数'), split_rows, 'compact')}
  {classification_tables(classification)}
  <p class="small">二値正解率: 要確認を適用する前の全画像で測定。採用率: review以外の割合。採用誤り率: 採用画像中の誤り。採用0件の誤り率は未定義。</p>
  <h2>testの二値混同行列</h2>{confusion_table(classification)}
  {figure(chart_image, '固定済み2方式の照明条件別test評価。', 'chart') if classification else '<p>比較図の実測入力は未測定。</p>'}
  <p class="notice">{esc(dark_note)}</p>
</section>''', f'''
<section class="page">
  <p class="kicker">2 / 物理搬送とAI接続</p><h1>独立episodeの搬送集計</h1>
  <p>{esc(state_note)}</p>
  {table(('指標', '実測値'), transfer_rows)}
  {table(('判定行先', '実測試行', '物理搬送成功', '成功件数 / 試行'), observed['destinations']) if observed['count'] else ''}
  <p><strong>成功判定:</strong> 部品全体が指定トレイ内にあり、床付近で、開放後に並進速度0.015 m/s未満・角速度0.15 rad/s未満を連続60物理step満たす。200 Hzで0.30秒。誤った良否行先はセル処理失敗とする。</p>
  <p>毎回、初期関節状態と部品位置・姿勢へresetする独立episode。20試行を実行しても、連続供給装置や20部品の蓄積を検証したことにはならない。</p>
  <p>3経路候補のデモ外観条件はvalから選び、固定条件の物理搬送とAI接続に使う。行先は再撮影画像から決める。この集計を分類accuracyのbenchmarkへ混ぜない。要確認搬送は保留であり、二値分類正解とは分けて記録する。</p>
  <h2>記録された失敗理由</h2>{failure_html}
  {figure(overview, 'カメラ視点だけのセル画像。ターミナルやデスクトップを含めない。', 'overview')}
</section>''', f'''
<section class="page">
  <p class="kicker">3 / 考察・出典・提出前確認</p><h1>分かったことと残作業</h1>
  <h2>画像分類資料の要約</h2><p>{esc(class_excerpt)}</p>
  <h2>搬送資料の要約</h2><p>{esc(result_excerpt)}</p>
  <h2>工夫と限界</h2>
  <p>小型CNNと明度しきい値を同じ画像で比較し、確信度による要確認を組み込んだ。分類APIは画像だけを受け取り、物理計測は搬送後の評価に使う。照明変化、レンダー残像、実機への汎化には未解決の制約がある。現実の検査性能は測定していない。</p>
  <p>native 2F-85用のarm-only URDFを用意し、静的USDのarm/mount変換と1e-6以内で整合を確認した。指先の形状からEE offsetを0.13 mに設定する。FK整合は接触・把持の成功保証ではない。</p>
  <h2>残作業</h2><ul>{''.join(f'<li>{esc(task)}</li>' for task in tasks)}</ul>
  <h2>出典・再現用成果物</h2>
  <p class="small">実測: docs/classification_metrics.json、artifacts/evaluation20/metrics.json。詳細: docs/CLASSIFICATION.md、docs/RESULTS.md。既存の搬送実験制御を必要な範囲でコピーし、今回の検査・AI・測定を追加した。NVIDIA由来の設定の著作権・ライセンス表示を保持する。</p>
  <p class="small">ロボットはNVIDIA Isaac Sim 6.0の組立済みUR10e/Robotiq 2F-85を実行時に参照する。USD・メッシュ・Isaac Sim本体は同梱しない。公式アセット一覧: <a href="https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/usd_assets_robots.html">NVIDIA Isaac Sim 6.0 Robot Assets</a>。</p>
  <p class="small draft">匿名draft。1〜3分の動画（倍速は明記）・public GitHub・PDFの確認とOmniCampus提出は本人が行う。個人情報・認証情報・業務データを掲載しない。</p>
</section>''']
    return '\n'.join(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classification-metrics', type=Path, default=ROOT / 'docs/classification_metrics.json')
    parser.add_argument('--transfer-metrics', type=Path, default=ROOT / 'artifacts/evaluation20/metrics.json')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'artifacts/report')
    parser.add_argument('--html-only', action='store_true')
    args = parser.parse_args()
    classification = load_optional(args.classification_metrics)
    transfers = load_optional(args.transfer_metrics)
    css = (ROOT / 'docs/report.css').read_text(encoding='utf-8')
    body = render_report(classification, transfers, ROOT)
    built = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    document = ('<!doctype html><html lang="ja"><head><meta charset="utf-8">'
                '<title>AI外観検査・仕分けセル - 匿名検証レポート下書き</title>'
                '<meta name="description" content="画像分類と物理搬送を分けた匿名の検証レポート">'
                f'<style>{css}</style></head><body>{body}'
                f'<div class="build-note">Build: {built}</div></body></html>')
    # Fail rather than silently emitting an unredacted private path in markup.
    if PRIVATE_PATH.search(document):
        raise ValueError('Report contains a private absolute path')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.out_dir / 'inspection_report.html'
    pdf_path = args.out_dir / 'inspection_report.pdf'
    html_path.write_text(document, encoding='utf-8')
    if not args.html_only:
        from weasyprint import HTML
        HTML(string=document).write_pdf(pdf_path)
    print('Report HTML saved.' if args.html_only else 'Report HTML and PDF saved.')
    print(f'Measured transfer trials: {len(measured_transfers(transfers))}')


if __name__ == '__main__':
    main()
