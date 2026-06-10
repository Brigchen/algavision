# -*- coding: utf-8 -*-
"""
cluster_review_gui.py — 无监督聚类结果审核 GUI（Tkinter）

用法（独立运行）：
    python cluster_review_gui.py

也可从 modler_v2 直接调用：
    from cluster_review_gui import run_cluster_review
    run_cluster_review(parent_root, annotator_instance, output_dir, done_callback)

流程：
    1. 选择无标签图像文件夹 + 聚类参数 → 运行 UnsupervisedAnnotator
    2. 簇列表 + 缩略图网格：浏览每个簇的代表图
    3. 为每个簇输入类名（或标记为"杂质/跳过"）
    4. 一键导出 → 类别子目录结构（可直接作为 modler_v2 的训练数据目录）

@author: brigchen@xmu.edu.cn
"""

import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

from unsupervised_annotator import UnsupervisedAnnotator
from models_registry import list_models

_THUMB_SIZE = 80     # 代表图缩略图尺寸
_GRID_COLS  = 8      # 缩略图网格列数
_MAX_THUMBS = 32     # 每簇最多显示张数


# ---------------------------------------------------------------------------
class ClusterReviewGUI:
    def __init__(self, root, done_callback=None):
        self.root = root
        self.root.title("无监督聚类标注")
        self.root.geometry("1200x760")
        self.done_callback = done_callback

        self.annotator    = None
        self.folder_path  = tk.StringVar(value="")
        self.output_path  = tk.StringVar(value="")
        self.model_var    = tk.StringVar(value="DINOv2-S")
        self.method_var   = tk.StringVar(value="hdbscan")
        self.min_size_var = tk.IntVar(value=10)
        self.batch_var    = tk.IntVar(value=32)
        self.recursive_var = tk.BooleanVar(value=True)

        # cluster_id -> StringVar（用户输入的类名）
        self.name_vars: dict[int, tk.StringVar] = {}
        # 已加载的 PhotoImage 引用（防 GC）
        self._thumb_refs = []

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        # 左：参数设置
        left = ttk.LabelFrame(top, text="参数设置", padding=6)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        _row = [0]
        def lr(label, widget_factory):
            ttk.Label(left, text=label).grid(row=_row[0], column=0, sticky=tk.W, pady=2)
            w = widget_factory(left)
            w.grid(row=_row[0], column=1, sticky=tk.EW, padx=4, pady=2)
            _row[0] += 1
            return w

        lr("嵌入骨干:", lambda p: ttk.Combobox(
            p, textvariable=self.model_var,
            values=list_models(), state="readonly", width=16))
        lr("聚类方法:", lambda p: ttk.Combobox(
            p, textvariable=self.method_var,
            values=["hdbscan", "kmeans"], state="readonly", width=16))
        lr("min_cluster_size:", lambda p: ttk.Spinbox(
            p, textvariable=self.min_size_var, from_=2, to=500, width=8))
        lr("批次大小:", lambda p: ttk.Spinbox(
            p, textvariable=self.batch_var, from_=4, to=256, width=8))

        recur_frame = ttk.Frame(left)
        recur_frame.grid(row=_row[0], column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(recur_frame, text="递归子目录",
                        variable=self.recursive_var).pack(anchor=tk.W)
        _row[0] += 1

        # 右：目录选择 + 按钮
        right = ttk.Frame(top, padding=6)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _dir_row(label, var, title):
            f = ttk.Frame(right); f.pack(fill=tk.X, pady=3)
            ttk.Label(f, text=label, width=10).pack(side=tk.LEFT)
            ttk.Entry(f, textvariable=var, width=45).pack(side=tk.LEFT, padx=4)
            ttk.Button(f, text="浏览…",
                       command=lambda: var.set(
                           filedialog.askdirectory(title=title) or var.get()
                       ), width=6).pack(side=tk.LEFT)

        _dir_row("图像目录:", self.folder_path, "选择无标签图像文件夹")
        _dir_row("输出目录:", self.output_path, "选择标注导出目录")

        btn_f = ttk.Frame(right); btn_f.pack(pady=6)
        ttk.Button(btn_f, text="▶ 开始聚类", width=14,
                   command=self._start_clustering).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_f, text="✔ 导出标注", width=14,
                   command=self._export).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_f, text="全部标记杂质", width=14,
                   command=self._mark_all_noise).pack(side=tk.LEFT, padx=4)

        # 进度条
        self.progress = ttk.Progressbar(right, length=400)
        self.progress.pack(pady=2)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(right, textvariable=self.status_var,
                  foreground="gray").pack(anchor=tk.W)

        # 主区域：左=簇列表，右=缩略图
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # 簇列表
        list_frame = ttk.LabelFrame(main, text="聚类列表", padding=4, width=280)
        main.add(list_frame, weight=1)

        cols = ("id", "数量", "类名")
        self.cluster_tree = ttk.Treeview(list_frame, columns=cols,
                                          show="headings", height=25)
        for c, w in zip(cols, (50, 60, 140)):
            self.cluster_tree.heading(c, text=c)
            self.cluster_tree.column(c, width=w, anchor=tk.CENTER)
        self.cluster_tree.tag_configure("noise", foreground="gray")
        self.cluster_tree.tag_configure("named", foreground="green4")
        vsb = ttk.Scrollbar(list_frame, command=self.cluster_tree.yview)
        self.cluster_tree.configure(yscrollcommand=vsb.set)
        self.cluster_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.cluster_tree.bind("<<TreeviewSelect>>", self._on_cluster_select)

        # 缩略图 + 命名区
        detail_frame = ttk.LabelFrame(main, text="代表图 & 命名", padding=6)
        main.add(detail_frame, weight=4)
        detail_frame.rowconfigure(0, weight=1)
        detail_frame.columnconfigure(0, weight=1)

        # 命名栏
        name_bar = ttk.Frame(detail_frame); name_bar.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(name_bar, text="本簇类名:").pack(side=tk.LEFT)
        self.name_entry = ttk.Entry(name_bar, width=24)
        self.name_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(name_bar, text="应用", command=self._apply_name).pack(side=tk.LEFT)
        ttk.Button(name_bar, text="标记为杂质",
                   command=self._mark_noise).pack(side=tk.LEFT, padx=6)
        self.cluster_info = ttk.Label(name_bar, text="", foreground="navy")
        self.cluster_info.pack(side=tk.LEFT, padx=10)

        # 缩略图滚动画布
        thumb_container = ttk.Frame(detail_frame)
        thumb_container.pack(fill=tk.BOTH, expand=True)
        thumb_container.rowconfigure(0, weight=1)
        thumb_container.columnconfigure(0, weight=1)

        self.thumb_canvas = tk.Canvas(thumb_container, bg="#f4f4f4",
                                       highlightthickness=0)
        vsb2 = ttk.Scrollbar(thumb_container, orient=tk.VERTICAL,
                              command=self.thumb_canvas.yview)
        self.thumb_canvas.configure(yscrollcommand=vsb2.set)
        self.thumb_canvas.grid(row=0, column=0, sticky="nsew")
        vsb2.grid(row=0, column=1, sticky="ns")

        self.thumb_frame = ttk.Frame(self.thumb_canvas)
        self.thumb_canvas.create_window((0, 0), window=self.thumb_frame, anchor="nw")
        self.thumb_frame.bind("<Configure>",
            lambda e: self.thumb_canvas.configure(
                scrollregion=self.thumb_canvas.bbox("all")))
        self.thumb_canvas.bind("<Enter>",
            lambda e: self.thumb_canvas.bind_all(
                "<MouseWheel>",
                lambda ev: self.thumb_canvas.yview_scroll(
                    int(-1 * ev.delta / 120), "units")))
        self.thumb_canvas.bind("<Leave>",
            lambda e: self.thumb_canvas.unbind_all("<MouseWheel>"))

    # ------------------------------------------------------------------
    def _log(self, msg, color="gray"):
        self.status_var.set(msg)
        self.root.update_idletasks()

    # ------------------------------------------------------------------
    def _start_clustering(self):
        folder = self.folder_path.get().strip()
        if not folder or not Path(folder).is_dir():
            messagebox.showwarning("提示", "请先选择有效的图像目录。")
            return

        self.progress["value"] = 0
        self.progress["maximum"] = 100
        self._log("准备聚类…")
        self.name_vars.clear()
        for item in self.cluster_tree.get_children():
            self.cluster_tree.delete(item)
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        def progress_cb(done, total):
            if total > 0:
                self.progress["maximum"] = total
                self.progress["value"] = done
                self.root.update_idletasks()

        def worker():
            try:
                self.annotator = UnsupervisedAnnotator(
                    model_key=self.model_var.get())
                self._log("加载模型中…")
                labels, reps, paths = self.annotator.run(
                    folder,
                    recursive=self.recursive_var.get(),
                    batch_size=self.batch_var.get(),
                    cluster_method=self.method_var.get(),
                    min_cluster_size=self.min_size_var.get(),
                    progress_cb=progress_cb,
                    log_cb=lambda m: self._log(m),
                )
                self.root.after(0, self._populate_list, labels, reps, paths)
            except Exception as e:
                import traceback; traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror("错误", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _populate_list(self, labels, reps, paths):
        unique = sorted(set(labels.tolist()))
        for cid in unique:
            count = int(np.sum(labels == cid))
            var = tk.StringVar(value=("__noise__" if cid == -1 else ""))
            self.name_vars[cid] = var
            tag = "noise" if cid == -1 else ""
            disp_name = "⚠ 噪声/杂质" if cid == -1 else ""
            self.cluster_tree.insert(
                "", tk.END,
                iid=str(cid),
                values=(cid if cid != -1 else "噪声", count, disp_name),
                tags=(tag,),
            )
        total = len(paths)
        n_noise = int(np.sum(labels == -1))
        n_cls = len([c for c in unique if c != -1])
        self._log(f"聚类完成: {total} 张图 → {n_cls} 个簇 + {n_noise} 噪声")
        self.progress["value"] = self.progress["maximum"]

    # ------------------------------------------------------------------
    def _on_cluster_select(self, _event):
        sel = self.cluster_tree.selection()
        if not sel:
            return
        cid = int(sel[0])
        if self.annotator is None:
            return
        count = int(np.sum(self.annotator.labels == cid))
        self.cluster_info.config(
            text=f"簇 {cid}  共 {count} 张"
            if cid != -1 else f"噪声/杂质  共 {count} 张")
        current = self.name_vars.get(cid, tk.StringVar()).get()
        self.name_entry.delete(0, tk.END)
        if current not in ("__noise__", ""):
            self.name_entry.insert(0, current)

        self._show_thumbnails(cid)

    def _show_thumbnails(self, cid):
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        if self.annotator is None:
            return
        rep_idxs = self.annotator.reps.get(cid, [])[:_MAX_THUMBS]
        col = 0
        for idx in rep_idxs:
            path = self.annotator.image_paths[idx]
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((_THUMB_SIZE, _THUMB_SIZE))
                photo = ImageTk.PhotoImage(img)
                self._thumb_refs.append(photo)
                lbl = tk.Label(self.thumb_frame, image=photo, bg="#f4f4f4",
                               relief=tk.FLAT, bd=1)
                row, c = divmod(col, _GRID_COLS)
                lbl.grid(row=row, column=c, padx=2, pady=2)
                col += 1
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _apply_name(self):
        sel = self.cluster_tree.selection()
        if not sel:
            return
        cid = int(sel[0])
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("提示", "请输入类名。")
            return
        self.name_vars[cid].set(name)
        self.cluster_tree.item(str(cid), values=(
            cid if cid != -1 else "噪声", 
            self.cluster_tree.item(str(cid))["values"][1],
            name,
        ), tags=("named",))
        self._log(f"簇 {cid} → '{name}'")

    def _mark_noise(self):
        sel = self.cluster_tree.selection()
        if not sel:
            return
        cid = int(sel[0])
        self.name_vars[cid].set("__noise__")
        self.cluster_tree.item(str(cid), values=(
            cid if cid != -1 else "噪声",
            self.cluster_tree.item(str(cid))["values"][1],
            "⚠ 杂质(跳过)",
        ), tags=("noise",))
        self.name_entry.delete(0, tk.END)

    def _mark_all_noise(self):
        if not self.name_vars:
            return
        for cid in self.name_vars:
            if cid == -1:
                continue
            self.name_vars[cid].set("__noise__")
            self.cluster_tree.item(str(cid), values=(
                cid,
                self.cluster_tree.item(str(cid))["values"][1],
                "⚠ 杂质(跳过)",
            ), tags=("noise",))
        self._log("所有簇已标记为杂质/跳过。")

    # ------------------------------------------------------------------
    def _export(self):
        if self.annotator is None or len(self.annotator.image_paths) == 0:
            messagebox.showwarning("提示", "请先完成聚类。")
            return

        named = {cid: v.get() for cid, v in self.name_vars.items()
                 if v.get() and v.get() != "__noise__"}
        if not named:
            messagebox.showwarning("提示", "请至少为一个簇命名后再导出。")
            return

        out_dir = self.output_path.get().strip()
        if not out_dir:
            out_dir = filedialog.askdirectory(title="选择标注导出目录")
        if not out_dir:
            return

        try:
            stats = self.annotator.export(
                out_dir, named, copy=True, include_noise=False)
            total = sum(stats.values())
            detail = "\n".join(f"  {k}: {v} 张" for k, v in stats.items())
            msg = f"导出完成！共 {total} 张图像\n{detail}\n\n导出目录:\n{out_dir}"
            messagebox.showinfo("导出成功", msg)
            self._log(f"已导出 {total} 张 → {out_dir}")
            if self.done_callback:
                self.done_callback(out_dir, stats)
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("导出失败", str(e))


# ---------------------------------------------------------------------------
def run_cluster_review(parent=None, annotator=None,
                       output_dir="", done_callback=None):
    """从 modler_v2 或其他 GUI 内弹出子窗口。"""
    win = tk.Toplevel(parent) if parent else tk.Tk()
    gui = ClusterReviewGUI(win, done_callback=done_callback)
    if annotator is not None:
        gui.annotator = annotator
    if output_dir:
        gui.output_path.set(output_dir)
    if parent is None:
        win.mainloop()
    return gui


# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    ClusterReviewGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
