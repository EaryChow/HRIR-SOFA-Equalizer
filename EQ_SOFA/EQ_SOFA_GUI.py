import sofar
import re
import numpy as np
from scipy import signal
import tkinter as tk
from tkinter import ttk, messagebox
import os
import platform
import subprocess
import pathlib
from autoeq.peq import LowShelf, HighShelf, Peaking

def design_eq(filt_type: str, fs: float, fc: float, gain_db: float, q: float) -> np.ndarray:
    # Dummy frequency array (not used for coefficient calculation)
    f = np.array([1000])

    if filt_type == 'LSC':
        filt = LowShelf(
            f=f, fs=fs, fc=fc, q=q, gain=gain_db,
            optimize_fc=False, optimize_q=False, optimize_gain=False
        )
    elif filt_type == 'HSC':
        filt = HighShelf(
            f=f, fs=fs, fc=fc, q=q, gain=gain_db,
            optimize_fc=False, optimize_q=False, optimize_gain=False
        )
    elif filt_type == 'PK':
        filt = Peaking(
            f=f, fs=fs, fc=fc, q=q, gain=gain_db,
            optimize_fc=False, optimize_q=False, optimize_gain=False
        )
    else:
        raise ValueError(f"Unsupported filter type: {filt_type}")

    # Get all coefficients from AutoEQ
    a0, a1, a2, b0, b1, b2 = filt.biquad_coefficients()

    return np.array([[b0, b1, b2, 1.0, -a1, -a2]])

# --------------------------
# Helper functions
# --------------------------
def parse_eq_file(eq_file):
    """Parse an EQ file to extract preamp gain and filters."""
    preamp_gain = 0.0
    filters = []
    filter_regex = re.compile(
        r'^Filter\s+\d+\s*:\s*ON\s+(LS|HS|PK)\s+Fc\s+([\d.]+)\s+Hz\s+Gain\s+([+-]?\d+\.?\d*)\s+dB(?:\s+Q\s+([\d.]+))?',
        re.IGNORECASE
    )

    with open(eq_file, 'r') as f:
        eq_lines = f.readlines()

    for line in eq_lines:
        clean_line = line.split('#')[0].strip()
        if not clean_line:
            continue

        if clean_line.lower().startswith('preamp:'):
            try:
                gain_str = re.search(r'[-+]?\d+\.?\d*', clean_line.split(':', 1)[1]).group()
                preamp_gain += float(gain_str)
            except (AttributeError, ValueError) as e:
                print(f"Warning: Could not parse preamp line: {clean_line}\nError: {e}")
            continue

        if clean_line.startswith('Filter'):
            if 'None' in clean_line or 'OFF' in clean_line.upper():
                continue

            match = filter_regex.match(clean_line)
            if not match:
                continue

            try:
                filt_type = match.group(1).upper()
                if filt_type == 'LS':
                    filt_type = 'LSC'
                elif filt_type == 'HS':
                    filt_type = 'HSC'

                fc = float(match.group(2))
                gain = float(match.group(3))
                q = match.group(4)

                if q is None and filt_type in ('LSC', 'HSC'):
                    q = 0.707
                else:
                    q = float(q) if q else 1.0

                filters.append((filt_type, fc, gain, q))
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not parse filter line: {clean_line}\nError: {e}")

    return preamp_gain, filters

def copy_sofa_structure(orig_sofa):
    """Create a new SOFA object with the same structure as the original."""
    new = sofar.Sofa(convention='SimpleFreeFieldHRIR')
    new.protected = False
    new.GLOBAL_SOFAConventions = 'SimpleFreeFieldHRIR'
    new.GLOBAL_SOFAConventionsVersion = '1.0'
    new.GLOBAL_APIName = 'sofar'
    new.GLOBAL_APIVersion = sofar.__version__
    new.GLOBAL_ApplicationName = 'EQ Modified HRIR'
    new.GLOBAL_DataType = 'FIR'
    new.GLOBAL_RoomType = 'free field'
    new.GLOBAL_DatabaseName = orig_sofa.GLOBAL_DatabaseName
    new.protected = True

    new.ListenerPosition = orig_sofa.ListenerPosition.copy()
    new.ReceiverPosition = orig_sofa.ReceiverPosition.copy()
    new.SourcePosition = orig_sofa.SourcePosition.copy()
    new.EmitterPosition = orig_sofa.EmitterPosition.copy()
    new.ListenerUp = orig_sofa.ListenerUp.copy()
    new.ListenerView = orig_sofa.ListenerView.copy()

    new.Data_IR = orig_sofa.Data_IR.copy()
    new.Data_SamplingRate = np.array([[orig_sofa.Data_SamplingRate]], dtype=np.float64)
    new.Data_Delay = orig_sofa.Data_Delay.copy()

    return new

# --------------------------
# Main processing workflows
# --------------------------
def process_files(input_sofa: str, eq_file: str, output_sofa: str):
    orig = sofar.read_sofa(input_sofa, verify=False)
    new = copy_sofa_structure(orig)

    preamp_gain, filters = parse_eq_file(eq_file)
    fs = int(new.Data_SamplingRate.item())

    # Design filters
    sos_list = []
    for filt in filters:
        filt_type, fc, gain, q = filt
        sos = design_eq(filt_type, fs, fc, gain, q)
        sos_list.append(sos)

    full_sos = np.vstack(sos_list) if sos_list else None

    # Apply processing
    preamp_linear = 10 ** (preamp_gain / 20.0)
    ir_data = new.Data_IR

    for m in range(ir_data.shape[0]):
        for r in range(ir_data.shape[1]):
            ir = ir_data[m, r, :].copy()
            ir *= preamp_linear
            if full_sos is not None:
                ir = signal.sosfilt(full_sos, ir)
            ir_data[m, r, :] = ir

    sofar.write_sofa(output_sofa, new)

def process_separate_files(input_sofa: str, left_eq: str, right_eq: str, output_sofa: str):
    orig = sofar.read_sofa(input_sofa, verify=False)
    new = copy_sofa_structure(orig)

    # Parse EQ files
    left_preamp, left_filters = parse_eq_file(left_eq)
    right_preamp, right_filters = parse_eq_file(right_eq)
    applied_preamp = min(left_preamp, right_preamp)

    fs = int(new.Data_SamplingRate.item())

    # Design left filters
    left_sos_list = []
    for filt in left_filters:
        filt_type, fc, gain, q = filt
        sos = design_eq(filt_type, fs, fc, gain, q)
        left_sos_list.append(sos)
    left_sos = np.vstack(left_sos_list) if left_sos_list else None

    # Design right filters
    right_sos_list = []
    for filt in right_filters:
        filt_type, fc, gain, q = filt
        sos = design_eq(filt_type, fs, fc, gain, q)
        right_sos_list.append(sos)
    right_sos = np.vstack(right_sos_list) if right_sos_list else None

    # Apply processing
    preamp_linear = 10 ** (applied_preamp / 20.0)
    ir_data = new.Data_IR

    for m in range(ir_data.shape[0]):
        for r in range(ir_data.shape[1]):
            ir = ir_data[m, r, :].copy()
            ir *= preamp_linear
            if r == 0 and left_sos is not None:
                ir = signal.sosfilt(left_sos, ir)
            elif r == 1 and right_sos is not None:
                ir = signal.sosfilt(right_sos, ir)
            ir_data[m, r, :] = ir

    sofar.write_sofa(output_sofa, new)

# --------------------------
# GUI Application
# --------------------------
class EQApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SOFA EQ Processor")
        self.geometry("500x400")

        os.makedirs('input', exist_ok=True)
        os.makedirs('output', exist_ok=True)

        self.selected_sofa = tk.StringVar()
        self.selected_eq = tk.StringVar()
        self.selected_left_eq = tk.StringVar()
        self.selected_right_eq = tk.StringVar()

        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.show_sofa_selection()

    def show_sofa_selection(self):
        for widget in self.container.winfo_children():
            widget.destroy()

        label = ttk.Label(self.container, text="Select SOFA File:")
        label.pack(pady=10)

        self.tree = ttk.Treeview(self.container, columns=('fullpath'), show='tree')
        self.tree.column('#0', width=400)
        self.tree.column('fullpath', width=0, stretch=tk.NO)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=20)

        # Populate SOFA files
        input_dir = pathlib.Path('input')
        sofa_files = list(input_dir.rglob('*.sofa'))

        dir_map = {}
        for file_path in sofa_files:
            relative = file_path.relative_to(input_dir)
            parts = relative.parts
            parent_iid = ''

            for i in range(len(parts)-1):
                dir_path = input_dir.joinpath(*parts[:i+1])
                if dir_path not in dir_map:
                    dir_iid = self.tree.insert(parent_iid, 'end', text=parts[i], open=False)
                    dir_map[dir_path] = dir_iid
                else:
                    dir_iid = dir_map[dir_path]
                parent_iid = dir_iid

            self.tree.insert(parent_iid, 'end', text=parts[-1], values=[str(file_path)])

        self.next_btn = ttk.Button(self.container, text="Next", command=self.ask_separate_eq, state=tk.DISABLED)
        self.next_btn.pack(pady=10)
        self.tree.bind('<<TreeviewSelect>>', self.on_sofa_select)

    def on_sofa_select(self, event):
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            if item['values']:
                self.next_btn.config(state=tk.NORMAL)
                self.selected_sofa.set(item['values'][0])
            else:
                self.next_btn.config(state=tk.DISABLED)
        else:
            self.next_btn.config(state=tk.DISABLED)

    def ask_separate_eq(self):
        for widget in self.container.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self.container)
        main_frame.pack(expand=True, pady=20)

        label = ttk.Label(main_frame, text="Use separate EQ for left and right channels?")
        label.pack(pady=(0, 20))

        yes_btn = ttk.Button(main_frame, text="Yes", command=self.show_left_eq_selection)
        yes_btn.pack(pady=10)

        no_frame = ttk.Frame(main_frame)
        no_frame.pack(pady=10)

        no_btn = ttk.Button(no_frame, text="No", command=self.show_eq_selection)
        no_btn.pack()

        explanation = ttk.Label(no_frame,
            text="Select No will use the same EQ on both channels,\nbasically EQing the binaural average response",
            foreground="#666666", # Linear sRGB value: 0.132891
            font=('TkDefaultFont', 9))
        explanation.pack(pady=(5, 0))

        self.container.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

    def show_left_eq_selection(self):
        for widget in self.container.winfo_children():
            widget.destroy()

        label = ttk.Label(self.container, text="Select Left Channel EQ File:")
        label.pack(pady=10)

        self.tree = ttk.Treeview(self.container, columns=('fullpath'), show='tree')
        self.tree.column('#0', width=400)
        self.tree.column('fullpath', width=0, stretch=tk.NO)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=20)

        # Populate EQ files
        input_dir = pathlib.Path('input')
        eq_files = list(input_dir.rglob('*.txt'))

        dir_map = {}
        for file_path in eq_files:
            relative = file_path.relative_to(input_dir)
            parts = relative.parts
            parent_iid = ''

            for i in range(len(parts)-1):
                dir_path = input_dir.joinpath(*parts[:i+1])
                if dir_path not in dir_map:
                    dir_iid = self.tree.insert(parent_iid, 'end', text=parts[i], open=False)
                    dir_map[dir_path] = dir_iid
                else:
                    dir_iid = dir_map[dir_path]
                parent_iid = dir_iid

            self.tree.insert(parent_iid, 'end', text=parts[-1], values=[str(file_path)])

        self.next_btn = ttk.Button(self.container, text="Next", command=self.show_right_eq_selection, state=tk.DISABLED)
        self.next_btn.pack(pady=10)
        self.tree.bind('<<TreeviewSelect>>', self.on_left_eq_select)

    def on_left_eq_select(self, event):
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            if item['values']:
                self.next_btn.config(state=tk.NORMAL)
                self.selected_left_eq.set(item['values'][0])
            else:
                self.next_btn.config(state=tk.DISABLED)
        else:
            self.next_btn.config(state=tk.DISABLED)

    def show_right_eq_selection(self):
        for widget in self.container.winfo_children():
            widget.destroy()

        label = ttk.Label(self.container, text="Select Right Channel EQ File:")
        label.pack(pady=10)

        self.tree = ttk.Treeview(self.container, columns=('fullpath'), show='tree')
        self.tree.column('#0', width=400)
        self.tree.column('fullpath', width=0, stretch=tk.NO)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=20)

        # Populate EQ files
        input_dir = pathlib.Path('input')
        eq_files = list(input_dir.rglob('*.txt'))

        dir_map = {}
        for file_path in eq_files:
            relative = file_path.relative_to(input_dir)
            parts = relative.parts
            parent_iid = ''

            for i in range(len(parts)-1):
                dir_path = input_dir.joinpath(*parts[:i+1])
                if dir_path not in dir_map:
                    dir_iid = self.tree.insert(parent_iid, 'end', text=parts[i], open=False)
                    dir_map[dir_path] = dir_iid
                else:
                    dir_iid = dir_map[dir_path]
                parent_iid = dir_iid

            self.tree.insert(parent_iid, 'end', text=parts[-1], values=[str(file_path)])

        self.process_btn = ttk.Button(self.container, text="Process", command=self.process_separate_files, state=tk.DISABLED)
        self.process_btn.pack(pady=10)
        self.tree.bind('<<TreeviewSelect>>', self.on_right_eq_select)

    def on_right_eq_select(self, event):
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            if item['values']:
                self.process_btn.config(state=tk.NORMAL)
                self.selected_right_eq.set(item['values'][0])
            else:
                self.process_btn.config(state=tk.DISABLED)
        else:
            self.process_btn.config(state=tk.DISABLED)

    def show_eq_selection(self):
        for widget in self.container.winfo_children():
            widget.destroy()

        label = ttk.Label(self.container, text="Select EQ File:")
        label.pack(pady=10)

        self.tree = ttk.Treeview(self.container, columns=('fullpath'), show='tree')
        self.tree.column('#0', width=400)
        self.tree.column('fullpath', width=0, stretch=tk.NO)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=20)

        # Populate EQ files
        input_dir = pathlib.Path('input')
        eq_files = list(input_dir.rglob('*.txt'))

        dir_map = {}
        for file_path in eq_files:
            relative = file_path.relative_to(input_dir)
            parts = relative.parts
            parent_iid = ''

            for i in range(len(parts)-1):
                dir_path = input_dir.joinpath(*parts[:i+1])
                if dir_path not in dir_map:
                    dir_iid = self.tree.insert(parent_iid, 'end', text=parts[i], open=False)
                    dir_map[dir_path] = dir_iid
                else:
                    dir_iid = dir_map[dir_path]
                parent_iid = dir_iid

            self.tree.insert(parent_iid, 'end', text=parts[-1], values=[str(file_path)])

        self.process_btn = ttk.Button(self.container, text="Process", command=self.process_files, state=tk.DISABLED)
        self.process_btn.pack(pady=10)
        self.tree.bind('<<TreeviewSelect>>', self.on_eq_select)

    def on_eq_select(self, event):
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            if item['values']:
                self.process_btn.config(state=tk.NORMAL)
                self.selected_eq.set(item['values'][0])
            else:
                self.process_btn.config(state=tk.DISABLED)
        else:
            self.process_btn.config(state=tk.DISABLED)

    def process_files(self):
        input_sofa = self.selected_sofa.get()
        eq_file = self.selected_eq.get()

        sofa_name = os.path.splitext(os.path.basename(input_sofa))[0]
        output_sofa = os.path.join('output', f"{sofa_name}_EQ_applied.sofa")

        try:
            process_files(input_sofa, eq_file, output_sofa)
            messagebox.showinfo("Success", f"Processing completed!\nOutput saved to:\n{output_sofa}")
            self.open_output_folder()
        except Exception as e:
            messagebox.showerror("Error", str(e))
        self.destroy()

    def process_separate_files(self):
        input_sofa = self.selected_sofa.get()
        left_eq_file = self.selected_left_eq.get()
        right_eq_file = self.selected_right_eq.get()

        sofa_name = os.path.splitext(os.path.basename(input_sofa))[0]
        output_sofa = os.path.join('output', f"{sofa_name}_EQ_applied.sofa")

        try:
            process_separate_files(input_sofa, left_eq_file, right_eq_file, output_sofa)
            messagebox.showinfo("Success", f"Processing completed!\nOutput saved to:\n{output_sofa}")
            self.open_output_folder()
        except Exception as e:
            messagebox.showerror("Error", str(e))
        self.destroy()

    def open_output_folder(self):
        try:
            output_path = os.path.abspath('output')
            if platform.system() == "Windows":
                os.startfile(output_path)
            elif platform.system() == "Darwin":
                subprocess.run(["open", output_path], check=True)
            else:
                subprocess.run(["xdg-open", output_path], check=True)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open output folder:\n{str(e)}")

if __name__ == "__main__":
    app = EQApp()
    app.mainloop()