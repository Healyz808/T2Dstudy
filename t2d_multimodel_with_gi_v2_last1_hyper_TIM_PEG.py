import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon

# ── 配置参数 ──────────────────────────────────────────────────────────────────
BASE_PATH  = "artifacts/test_custom_seeds/run_seed_29"
OUTPUT_DIR = "artifacts/test_custom_seeds/PEG_results_seed_29"

# MODELS         = ["lstm", "gru", "tcn", "transformer"]
# PH_LIST        = [15, 30, 45, 60, 120]
MODELS         = ["transformer"]
PH_LIST        = [45, 60, 120]
TARGET_FEATURE = "glucose_carbs_gi_meal_effects"
LIMS           = 550

ZONE_COLORS = {'A':'#c8f0d0','B':'#aed6f1','C':'#fde8a0','D':'#f1a8a5','E':'#d98fcc'}

# ══════════════════════════════════════════════════════════════════════════════
# 单位还原：数据存储时已按 scale 归一化，需除以 scale 还原为 mg/dL
# scale 值从每个 results.json 的顶层 'scale' 字段读取
# ══════════════════════════════════════════════════════════════════════════════

# ── Parkes EGA T2D 分类（参考脚本 parkes_type_2 逻辑）────────────────────────
def _above(x1, y1, x2, y2, act, pred):
    if x1 == x2: return False
    y_line = ((y1 - y2) * act + y2 * x1 - y1 * x2) / (x1 - x2)
    return pred >= y_line

def _below(x1, y1, x2, y2, act, pred):
    return not _above(x1, y1, x2, y2, act, pred)

def classify_zone_t2d(act, pred):
    # Zone E
    if pred > 200 and _above(35, 200, 50, 550, act, pred):
        return 'E'
    # Zone D - left upper
    if (pred > 80
            and _above(25,  80,  35,  90, act, pred)
            and _above(35,  90, 125, 550, act, pred)):
        return 'D'
    # Zone D - right lower
    if (act > 250
            and _below(250,  40, 410, 110, act, pred)
            and _below(410, 110, 550, 160, act, pred)):
        return 'D'
    # Zone C - left upper
    if pred > 60 and _above(30, 60, 280, 550, act, pred):
        return 'C'
    # Zone C - right lower
    if (_below(90,   0, 260, 130, act, pred)
            and _below(260, 130, 550, 250, act, pred)):
        return 'C'
    # Zone B - left upper
    if (pred > 50
            and _above(30,  50, 230, 330, act, pred)
            and (act < 230 or _above(230, 330, 440, 550, act, pred))):
        return 'B'
    # Zone B - right lower
    if (act > 50
            and _below( 50,  30,  90,  80, act, pred)
            and _below( 90,  80, 330, 230, act, pred)
            and (act < 330 or _below(330, 230, 550, 450, act, pred))):
        return 'B'
    return 'A'

def classify_zones_t2d(ref, pred):
    return [classify_zone_t2d(r, p) for r, p in zip(ref, pred)]

def compute_zone_stats(ref, pred):
    if len(ref) == 0:
        return {'Zone A%': float('nan'), 'Zone A+B%': float('nan')}
    zones = classify_zones_t2d(ref, pred)
    total = len(zones)
    a = zones.count('A'); b = zones.count('B')
    return {'Zone A%':   round(a / total * 100, 2),
            'Zone A+B%': round((a + b) / total * 100, 2)}


# ── 区域填色 ──────────────────────────────────────────────────────────────────
def draw_peg_zones(ax):
    L = LIMS
    def fill(pts, color, alpha=0.55, zorder=0):
        ax.add_patch(Polygon(np.clip(np.array(pts, float), 0, L),
                             closed=True, facecolor=color,
                             alpha=alpha, linewidth=0, zorder=zorder))

    # E
    fill([(0,200),(0,L),(50,L),(35,200)],              ZONE_COLORS['E'], zorder=0)
    fill([(250,0),(L,0),(L,L),(550,160),(410,110),(250,40)], ZONE_COLORS['E'], zorder=0)

    # D
    fill([(0,80),(25,80),(35,90),(125,L),(50,L),(35,200),(0,200)],
         ZONE_COLORS['D'], zorder=1)
    fill([(250,40),(410,110),(550,160),(550,250),(260,130),(90,0),(250,0)],
         ZONE_COLORS['D'], zorder=1)

    # C
    fill([(0,60),(30,60),(280,L),(125,L),(35,90),(25,80),(0,80)],
         ZONE_COLORS['C'], zorder=2)
    fill([(90,0),(260,130),(550,250),(550,450),(330,230),(90,80),(50,30),(50,0)],
         ZONE_COLORS['C'], zorder=2)

    # B
    fill([(0,50),(30,50),(230,330),(440,L),(280,L),(30,60),(0,60)],
         ZONE_COLORS['B'], alpha=0.50, zorder=3)
    fill([(0,0),(50,0),(50,30),(90,80),(330,230),(550,450),(L,L),
          (440,L),(230,330),(30,50),(0,50)],
         ZONE_COLORS['B'], alpha=0.50, zorder=3)

    # A（最上层）
    fill([(0,0),(0,50),(30,50),(230,330),(440,L),(L,L),(L,0)],
         ZONE_COLORS['A'], alpha=0.60, zorder=4)
    fill([(0,0),(50,0),(50,30),(90,80),(330,230),(550,450),(L,L),(0,0)],
         ZONE_COLORS['A'], alpha=0.60, zorder=4)


def draw_peg_boundaries(ax):
    kw = dict(color='black', lw=1.0, alpha=0.70, zorder=10)
    ax.plot([0,LIMS],[0,LIMS],'k--',lw=0.8,alpha=0.35,zorder=9)

    def draw(pts):
        pts = np.array(pts)
        ax.plot(pts[:,0], pts[:,1], **kw)

    draw([(0,50), (30,50), (230,330),(440,550)])          # B 上界
    draw([(0,60), (30,60), (280,550)])                    # C 上界
    draw([(0,80), (25,80), (35,90),  (125,550)])          # D 上界
    draw([(0,200),(35,200),(50,550)])                      # E 上界
    draw([(50,30),(90,80), (330,230),(550,450)])           # B 下界
    draw([(90,0), (260,130),(550,250)])                    # C 下界
    draw([(250,40),(410,110),(550,160)])                   # D 下界

    # 区域字母标注：对照参考图布局
    # 上方: E D C B A 沿左上→右下斜线
    # 下方: B C D 关于对角线对称
    # x 最小值 ≥ 30，避免被 ylabel 裁切
    labels = [
        # ── 上方（pred > ref 侧），E/D 往内移确保在绘图区内 ──
        ('E',  30, 480),   # 左上，y 留余量避免顶部裁切
        ('D',  68, 480),   # E 右侧同高
        ('C', 155, 455),
        ('B', 275, 415),
        ('A', 405, 380),
        # ── 下方（pred < ref 侧），D 往左移避免右边裁切 ──
        ('B', 420, 215),
        ('C', 450,  95),
        ('D', 480,  40),
    ]
    for letter, x, y in labels:
        ax.text(x, y, letter, fontsize=14, fontweight='bold',
                ha='center', va='center', color='#222222',
                alpha=0.72, zorder=12,
                bbox=dict(boxstyle='round,pad=0.18', fc='white',
                          ec='none', alpha=0.60))


# ── 数据读取 ──────────────────────────────────────────────────────────────────
def get_subjects(base_path):
    return sorted([d for d in os.listdir(base_path)
                   if os.path.isdir(os.path.join(base_path, d))])

def load_subject(base_path, sub, model, ph):
    """
    从 predictions 列表读取 y_true / y_pred。
    单位还原：数据以 scale 归一化存储，还原公式为 value / scale → mg/dL。
    scale 从 results.json 顶层 'scale' 字段读取，默认 0.01。
    """
    json_path = os.path.join(base_path, sub, model,
                             TARGET_FEATURE, f"PH{ph}", "results.json")
    if not os.path.exists(json_path):
        return [], []
    y_true, y_pred = [], []
    try:
        with open(json_path) as f:
            data = json.load(f)

        scale = float(data.get('scale', 0.01))   # 读取 scale，默认 0.01

        for entry in data.get('predictions', []):
            yt = entry.get('y_true', [])
            yp = entry.get('y_pred', [])
            if yt and isinstance(yt[0], list): yt = [x[0] for x in yt]
            if yp and isinstance(yp[0], list): yp = [x[0] for x in yp]
            # 除以 scale 还原为 mg/dL
            y_true.extend([v / scale for v in yt])
            y_pred.extend([v / scale for v in yp])
    except Exception as e:
        print(f"    [WARN] {sub}/{model}/PH{ph}: {e}")
    return y_true, y_pred

def load_all(base_path, model, ph):
    all_y_true, all_y_pred, count = [], [], 0
    for sub in get_subjects(base_path):
        yt, yp = load_subject(base_path, sub, model, ph)
        if yt:
            all_y_true.extend(yt); all_y_pred.extend(yp); count += 1
            print(f"  + {sub}  ({len(yt)} pts, "
                  f"ref {min(yt):.0f}~{max(yt):.0f} mg/dL)")
        else:
            tried = os.path.join(base_path, sub, model,
                                 TARGET_FEATURE, f"PH{ph}", "results.json")
            print(f"  - {sub}: no data  [tried: {tried}]")
    return all_y_true, all_y_pred, count


# ── 单图绘制 ──────────────────────────────────────────────────────────────────
def parkes_error_grid(ref, pred, title, ax, n_subjects=0):
    ref  = np.array(ref,  dtype=float)
    pred = np.array(pred, dtype=float)

    draw_peg_zones(ax)
    draw_peg_boundaries(ax)

    # 按 Zone 着色散点（与参考图一致）
    zones  = classify_zones_t2d(ref, pred)
    colors = {'A':'#27ae60','B':'#2980b9','C':'#e67e22','D':'#c0392b','E':'#8e44ad'}
    for zone in 'ABCDE':
        mask = np.array([z == zone for z in zones])
        if mask.any():
            ax.scatter(ref[mask], pred[mask], marker='o',
                       color=colors[zone], s=8, alpha=0.40,
                       linewidths=0, zorder=20, label=zone)

    ax.set_xlim(0, LIMS); ax.set_ylim(0, LIMS)
    ax.set_xlabel("Reference Glucose (mg/dL)", fontsize=14)
    ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=14)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=6)
    ax.set_aspect('equal'); ax.tick_params(labelsize=14)

    if len(ref) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='gray')
        return

    total  = len(zones)
    counts = {z: zones.count(z) for z in 'ABCDE'}
    # ── 图内统计文字（如需显示请取消注释）──
    # stats  = "\n".join(f"Zone {z}: {counts[z]/total*100:.1f}%"
    #                    for z in 'ABCDE' if counts[z] > 0)
    # ax.text(0.98, 0.02, stats, transform=ax.transAxes, fontsize=7,
    #         va='bottom', ha='right', zorder=25,
    #         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.9))
    #
    # rmse = np.sqrt(np.mean((pred - ref)**2))
    # mard = np.mean(np.abs(pred - ref) / np.clip(ref, 1, None)) * 100
    # ab   = (counts['A'] + counts['B']) / total * 100
    # info = (f"n={total:,}  subjects={n_subjects}\n"
    #         f"RMSE={rmse:.1f} mg/dL  MARD={mard:.1f}%\n"
    #         f"A+B={ab:.1f}%")
    # ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=7,
    #         va='top', ha='left', color='#333', zorder=25,
    #         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.9))


# ── 主流程 ────────────────────────────────────────────────────────────────────
def load_and_plot():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subjects = get_subjects(BASE_PATH)
    print(f"Found {len(subjects)} subjects: {subjects}\n")

    fig, axes = plt.subplots(len(MODELS), len(PH_LIST),
                             figsize=(5.5 * len(PH_LIST), 5.5 * len(MODELS)),
                             facecolor='white')
    if len(MODELS) == 1:  axes = [axes]
    if len(PH_LIST) == 1: axes = [[ax] for ax in axes]

    for i, model in enumerate(MODELS):
        for j, ph in enumerate(PH_LIST):
            print(f"{'='*45}\n  {model.upper()}  PH{ph}\n{'='*45}")
            y_true, y_pred, n_sub = load_all(BASE_PATH, model, ph)
            ax    = axes[i][j]
            title = f"PH {ph} min"
            if y_true:
                parkes_error_grid(y_true, y_pred, title, ax, n_subjects=n_sub)
            else:
                ax.set_title(f"{title}\n(No Data)", fontsize=14, color='gray')
                ax.set_xlim(0,LIMS); ax.set_ylim(0,LIMS); ax.set_aspect('equal')
                ax.text(0.5,0.5,"No data found",transform=ax.transAxes,
                        ha='center',va='center',fontsize=14,color='gray')

    legend_patches = [
        mpatches.Patch(fc='#2ecc71', alpha=0.8, label='Zone A - No clinical risk'),
        mpatches.Patch(fc='#3498db', alpha=0.8, label='Zone B - Little/no effect on clinical action'),
        mpatches.Patch(fc='#f39c12', alpha=0.8, label='Zone C - Overcorrection possible'),
        mpatches.Patch(fc='#e74c3c', alpha=0.8, label='Zone D - Failure to detect hypo/hyperglycemia'),
        mpatches.Patch(fc='#8e44ad', alpha=0.8, label='Zone E - Erroneous treatment'),
    ]
    fig.legend(handles=legend_patches, loc='upper center', ncol=2,
               fontsize=14, frameon=True, framealpha=0.92,
               bbox_to_anchor=(0.5, 1.25))
    # fig.suptitle(
    #     f"Parkes (Consensus) Error Grid  |  T2D  |  Seed 29  |  {TARGET_FEATURE}",
    #     fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.055, 1, 1])

    img_out = os.path.join(OUTPUT_DIR, "PEG_seed29_gi_effects.png")
    plt.savefig(img_out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nPEG image saved: {img_out}")
    plt.show()

    # 统计表格
    col_tuples = [(model.upper(), f"PH{ph}", metric)
                  for model in MODELS for ph in PH_LIST
                  for metric in ["Zone A%", "Zone A+B%"]]
    rows = {}
    for sub in subjects:
        row = {}
        for model in MODELS:
            for ph in PH_LIST:
                yt, yp = load_subject(BASE_PATH, sub, model, ph)
                stats  = compute_zone_stats(yt, yp)
                row[(model.upper(), f"PH{ph}", "Zone A%")]   = stats["Zone A%"]
                row[(model.upper(), f"PH{ph}", "Zone A+B%")] = stats["Zone A+B%"]
        rows[sub] = row

    df = pd.DataFrame(rows).T
    df.index.name = "Subject"
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Model","PH","Metric"])
    df = df.reindex(columns=pd.MultiIndex.from_tuples(
        col_tuples, names=["Model","PH","Metric"]))
    mean_row = df.mean(numeric_only=True).to_frame().T
    mean_row.index = ["Mean (all subjects)"]; mean_row.index.name = "Subject"
    df_out = pd.concat([df, mean_row])

    df_flat = df_out.copy()
    df_flat.columns = [f"{m}_{p}_{met}" for m,p,met in df_flat.columns]
    csv_out = os.path.join(OUTPUT_DIR, "PEG_zone_stats_seed29.csv")
    df_flat.to_csv(csv_out, float_format="%.2f")
    print(f"Zone stats CSV saved: {csv_out}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("\n-- Table preview --")
    print(df_out.to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    load_and_plot()

