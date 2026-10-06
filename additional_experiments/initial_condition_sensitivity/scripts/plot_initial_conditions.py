"""Validate compact diagnostics and plot initial-condition sensitivity."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lohe-additional-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from figure_style import apply_paper_style, format_axes

GROUPS = [("A", None), ("B", None), ("C", .05), ("C", .15), ("C", .30)]
LABELS = ["A", "B", "C\n0.05", "C\n0.15", "C\n0.30"]


def select(rows, dim, group, amp):
    return [r for r in rows if r["dimension"] == dim and r["group"] == group and r["amplitude"] == amp]


def finite(rows, key):
    return np.array([r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])], dtype=float)


def summarize(rows):
    summaries = []
    baseline = {(r["dimension"],r["K"],r.get("sigma_omega")):r for r in rows if r["group"]=="A"}
    for dim in (3, 5):
        for group, amp in GROUPS:
            selected = select(rows, dim, group, amp)
            summary = {"dimension": dim, "group": group, "amplitude": amp,
                       "case_count": len(selected),
                       "threshold_reached_count": sum(r["threshold_reached"] for r in selected),
                       "threshold_maintained_count": sum(r["threshold_maintained"] for r in selected),
                       "metric_valid_count": sum(r["metric_valid"] for r in selected)}
            for key in ("K_t_f_num", "rate_ratio", "relative_rate_error", "fit_r_squared",
                        "fit_count", "sphere_error_max", "initial_error", "initial_phi_min", "initial_phi_max",
                        "p_relative_L2_error", "polar_logtan_relative_L2_error", "max_phi_error"):
                values = finite(selected, key)
                for suffix, function in (("min", np.min), ("mean", np.mean), ("max", np.max)):
                    summary[f"{key}_{suffix}"] = float(function(values)) if values.size else None
            shifts = np.array([r["phi_bar_tf"]-r["initial_phi_bar"] for r in selected if r["phi_bar_tf"] is not None])
            summary["fast_phi_change_min"] = float(shifts.min()) if shifts.size else None
            summary["fast_phi_change_max"] = float(shifts.max()) if shifts.size else None
            differences = [abs(r["rate_ratio"]-baseline[(dim,r["K"],r.get("sigma_omega"))]["rate_ratio"])
                           for r in selected if r["rate_ratio"] is not None]
            summary["max_abs_rate_ratio_change_from_A"] = max(differences) if differences else None
            summary["initial_error_over_rho_min"] = min(r["initial_error"]/r["rho"] for r in selected)
            summary["initial_error_over_rho_max"] = max(r["initial_error"]/r["rho"] for r in selected)
            summaries.append(summary)
    with (OUT / "data/group_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (OUT / "data/group_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def save(fig, name):
    for extension in ("pdf", "png"):
        path = OUT / "figures" / f"{name}.{extension}"
        fig.savefig(path, bbox_inches="tight", dpi=600)
        print("Saved", path, flush=True)
    plt.close(fig)


def style():
    apply_paper_style()
    plt.rcParams.update({"font.size":17, "axes.labelsize":19, "xtick.labelsize":15,
                         "ytick.labelsize":15, "legend.fontsize":14})


def rate_plot(rows):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.5))
    fig.subplots_adjust(left=.12, right=.985, bottom=.30, top=.94, wspace=.40)
    for ax, dim, panel in zip(axes, (3,5), ("(a)","(b)")):
        plotted = []
        for position, (group, amp) in enumerate(GROUPS):
            selected = [r for r in select(rows,dim,group,amp) if r["metric_valid"]]
            values = finite(selected,"rate_ratio")
            if not values.size:
                continue
            jitter = np.linspace(-.13,.13,len(values))
            ax.scatter(position+jitter,values,s=25,color="#0072B2",alpha=.70,
                       edgecolors="none",zorder=5,label="Runs" if position==0 else None)
            ax.scatter([position],[np.median(values)],marker="_",s=180,color="#D55E00",
                       linewidths=2.2,zorder=6,label="Median" if position==0 else None)
            plotted.extend(values)
        ax.axhline(1.,color="black",linestyle="--",linewidth=2.2,zorder=3)
        # Use the same ratio scale in both dimensions for direct comparison.
        low, high = min(plotted+[1.]), max(plotted+[1.])
        if .999 <= low <= high <= 1.0014:
            ax.set_ylim(.999,1.0014)
            ax.set_yticks([.999,.9995,1.,1.0005,1.001])
        else:
            width = max(high-low,1e-4)
            ax.set_ylim(low-.15*width,high+.38*width)
        ax.set_xlim(-.4,4.4)
        ax.set_xticks(range(5),LABELS)
        ax.set_xlabel("Initial-condition group")
        ax.set_ylabel(r"$D_{\mathrm{meas}}/\Lambda_K$" if dim==3 else r"$m_{\mathrm{meas}}/m_{\mathrm{pred}}$")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        format_axes(ax)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.text(.5,-.35,panel,transform=ax.transAxes,ha="center",va="top",fontsize=17)
        ax.legend(loc="upper left",ncol=2,handlelength=.8,columnspacing=.4,handletextpad=.2,fontsize=14)
    save(fig,"fig_slow_rate_sensitivity")


def transient_plot(rows):
    style()
    fig, axes = plt.subplots(1,2,figsize=(7.8,4.5))
    fig.subplots_adjust(left=.11,right=.985,bottom=.28,top=.94,wspace=.30)
    colors = ["black","#D55E00","#0072B2","#009E73","#CC79A7"]
    for ax,dim,panel in zip(axes,(3,5),("(a)","(b)")):
        for (group,amp),color in zip(GROUPS,colors):
            selected = [r for r in select(rows,dim,group,amp) if r["K"]==(10. if dim==3 else 12.)
                        and (dim==5 or r["sigma_omega"]==.20)]
            curves=[]
            for row in selected:
                with np.load(OUT/"data/diagnostics"/f"{row['case_id']}.npz") as z:
                    t=z["ts"]*row["K"]
                    error=z["E_lock"] if dim==3 else np.maximum(z["R_ans"],z["E_phi"])
                    curves.append(error/row["rho"]**2)
            if not curves:
                continue
            values=np.stack(curves)
            # Retain the next saved sample so line segments reach the axes boundary.
            stop=min(len(t),int(np.searchsorted(t,12.,side="right"))+1)
            mask=np.arange(len(t))<stop
            label=group if amp is None else rf"C, $a={amp:.2f}$"
            ax.semilogy(t[mask],np.median(values,axis=0)[mask],color=color,lw=2.,label=label)
            if len(curves)>1:
                ax.fill_between(t[mask],values.min(axis=0)[mask],values.max(axis=0)[mask],color=color,alpha=.15)
        ax.axhline(5.,color="0.4",linestyle="--",linewidth=1.5)
        ax.set(xlim=(0,12),xlabel=r"$Kt$",ylabel=r"$E_{\mathrm{lock}}/\rho^2$" if dim==3 else r"$\max(R_{\mathrm{ans}},E_\phi)/\rho^2$")
        format_axes(ax)
        ax.legend(loc="upper right",fontsize=11)
        ax.text(.5,-.25,panel,transform=ax.transAxes,ha="center",va="top",fontsize=17)
    save(fig,"fig_fast_transient")



def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Experiment directory containing data/ and figures/.")
    args = parser.parse_args()
    if args.output_dir:
        OUT = args.output_dir.resolve()
    import run_initial_conditions as experiment
    experiment.OUT = OUT
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    rows, _ = experiment.validate()
    metadata = json.loads((OUT / "data/metadata.json").read_text())
    assert rows == metadata["cases"]
    summaries = summarize(rows)
    rate_plot(rows)
    transient_plot(rows)
    print("Saved group summaries and two sensitivity figures.", flush=True)


if __name__ == "__main__":
    main()
