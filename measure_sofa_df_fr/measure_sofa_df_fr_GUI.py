import sofar
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import csv
import os


class AppController:
    """Main application controller managing GUI flow and data processing"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HRIR Analysis Tool")
        self.root.geometry("600x400")
        self.current_frame = None
        self.selected_sofa = None
        self.export_formats = []
        self.reference_file = None
        self.csv_selections = []
        self.ref_align_freq = 500.0
        self.setup_frames()
        self.show_frame("FileSelect")

    def setup_frames(self):
        """Initialize all application frames"""
        self.frames = {
            "FileSelect": FileSelectFrame(self.root, self),
            "ExportFormat": ExportFormatFrame(self.root, self),
            "ReferenceSelect": ReferenceSelectFrame(self.root, self),
            "CSVOptions": CSVOptionsFrame(self.root, self)
        }

    def show_frame(self, name):
        """Show specified frame"""
        if self.current_frame:
            self.current_frame.pack_forget()
        self.current_frame = self.frames[name]
        self.current_frame.pack(expand=True, fill=tk.BOTH)
        self.current_frame.refresh()

    def start_export(self):
        """Handle the final export process"""
        try:
            process_data(self)
            output_dir = Path("output") / self.selected_sofa.stem
            subprocess.Popen(f'explorer "{output_dir.resolve()}"')
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
            self.root.destroy()


class BaseFrame(ttk.Frame):
    """Base class for application frames"""

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

    def refresh(self):
        """Update frame contents when shown"""
        pass


class FileSelectFrame(BaseFrame):
    """SOFA file selection frame with Treeview for directories"""

    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.sofa_files = []
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self, text="Select SOFA File").pack(pady=10)
        self.tree = ttk.Treeview(self, columns=('fullpath'), show='tree')
        self.tree.column('#0', width=400)
        self.tree.column('fullpath', width=0, stretch=tk.NO)
        self.tree.pack(expand=True, fill=tk.BOTH, padx=20)

        self.next_btn = ttk.Button(self, text="Next",
                                   command=lambda: self.controller.show_frame("ExportFormat"),
                                   state=tk.DISABLED)
        self.next_btn.pack(pady=10)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

    def refresh(self):
        input_dir = Path("input")
        self.tree.delete(*self.tree.get_children())
        sofa_files = list(input_dir.rglob("*.sofa"))

        if not sofa_files:
            messagebox.showerror("Error", "No SOFA files found in input folder or subfolders")
            self.controller.root.destroy()
            return

        dir_map = {}  # Maps directory Paths to Treeview item IDs
        for file_path in sofa_files:
            relative = file_path.relative_to(input_dir)
            parts = relative.parts
            parent_iid = ''

            # Build directory nodes
            for i in range(len(parts)-1):
                dir_path = input_dir.joinpath(*parts[:i+1])
                if dir_path not in dir_map:
                    dir_iid = self.tree.insert(parent_iid, 'end', text=parts[i], open=False)
                    dir_map[dir_path] = dir_iid
                else:
                    dir_iid = dir_map[dir_path]
                parent_iid = dir_iid

            # Add file node
            self.tree.insert(parent_iid, 'end', text=parts[-1], values=[str(file_path)])

    def on_select(self, event):
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            if item['values']:  # File selected
                self.controller.selected_sofa = Path(item['values'][0])
                self.next_btn.state(['!disabled'])
            else:  # Directory selected
                self.next_btn.state(['disabled'])
        else:
            self.next_btn.state(['disabled'])


class ExportFormatFrame(BaseFrame):
    """Export format selection frame"""

    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_widgets()

    def create_widgets(self):
        """Create frame components"""
        ttk.Label(self, text="Select Export Formats").pack(pady=10)

        self.png_var = tk.BooleanVar()
        self.csv_var = tk.BooleanVar()

        ttk.Checkbutton(self, text="PNG", variable=self.png_var,
                        command=self.update_state).pack(anchor=tk.W, padx=20)
        ttk.Checkbutton(self, text="CSV", variable=self.csv_var,
                        command=self.update_state).pack(anchor=tk.W, padx=20)

        self.next_btn = ttk.Button(self, text="Next", command=self.next_step,
                                   state=tk.DISABLED)
        self.next_btn.pack(pady=10)

        ttk.Button(self, text="Back",
                   command=lambda: self.controller.show_frame("FileSelect")).pack()

    def update_state(self):
        """Update button state based on selections"""
        if self.png_var.get() or self.csv_var.get():
            self.next_btn.state(['!disabled'])
        else:
            self.next_btn.state(['disabled'])

    def next_step(self):
        """Proceed to next appropriate frame"""
        self.controller.export_formats = []
        if self.png_var.get():
            self.controller.export_formats.append("png")
        if self.csv_var.get():
            self.controller.export_formats.append("csv")

        if "png" in self.controller.export_formats:
            self.controller.show_frame("ReferenceSelect")
        elif "csv" in self.controller.export_formats:
            self.controller.show_frame("CSVOptions")


class ReferenceSelectFrame(BaseFrame):
    """Reference curve selection frame with Treeview for directories"""

    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self, text="Select Reference Curve (Optional)").pack(pady=10)
        ttk.Label(self, text="overlay a reference curve in the plot for visual reference").pack()
        ttk.Label(self, text="Supported formats: TXT, CSV").pack()

        self.tree = ttk.Treeview(self, columns=('fullpath'), show='tree')
        self.tree.column('#0', width=400)
        self.tree.column('fullpath', width=0, stretch=tk.NO)
        self.tree.pack(expand=True, fill=tk.BOTH, padx=20)

        # Frequency alignment input
        input_frame = ttk.Frame(self)
        input_frame.pack(pady=10)

        ttk.Label(input_frame, text="Alignment Frequency (Hz):").pack(side=tk.LEFT, padx=(0, 10))
        self.freq_entry = ttk.Entry(input_frame, width=10)
        self.freq_entry.pack(side=tk.LEFT)
        self.freq_entry.insert(0, "500")  # Default value

        # Validate numeric input
        def validate_num(new_val):
            return new_val == "" or new_val.replace('.', '', 1).isdigit()

        vcmd = (self.register(validate_num), '%P')
        self.freq_entry.config(validate="key", validatecommand=vcmd)

        ttk.Label(input_frame,
                  text="Frequency where the reference overlaps the binaural average."
                       "\nDone by shifting the reference curve up or down"
                       "\nDefault is 500 Hz.").pack(
            side=tk.LEFT, padx=(10, 0))

        self.next_btn = ttk.Button(self, text="Next", command=self.next_step)
        self.next_btn.pack(pady=10)
        ttk.Button(self, text="Back",
                   command=lambda: self.controller.show_frame("ExportFormat")).pack()
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

    def refresh(self):
        input_dir = Path("input")
        self.tree.delete(*self.tree.get_children())
        ref_files = list(input_dir.rglob("*.[tT][xX][tT]")) + list(input_dir.rglob("*.[cC][sS][vV]"))

        dir_map = {}
        for file_path in ref_files:
            relative = file_path.relative_to(input_dir)
            parts = relative.parts
            parent_iid = ''

            # Build directory nodes
            for i in range(len(parts)-1):
                dir_path = input_dir.joinpath(*parts[:i+1])
                if dir_path not in dir_map:
                    dir_iid = self.tree.insert(parent_iid, 'end', text=parts[i], open=False)
                    dir_map[dir_path] = dir_iid
                else:
                    dir_iid = dir_map[dir_path]
                parent_iid = dir_iid

            # Add file node
            self.tree.insert(parent_iid, 'end', text=parts[-1], values=[str(file_path)])

    def on_select(self, event):
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            self.controller.reference_file = Path(item['values'][0]) if item['values'] else None
        else:
            self.controller.reference_file = None

    def next_step(self):
        """Navigate to next appropriate frame"""
        # Get alignment frequency
        freq_input = self.freq_entry.get()
        try:
            self.controller.ref_align_freq = float(freq_input)
        except ValueError:
            self.controller.ref_align_freq = 500.0  # Default to 500 if invalid

        if "csv" in self.controller.export_formats:
            self.controller.show_frame("CSVOptions")
        else:
            self.controller.csv_selections = []
            self.controller.start_export()


class CSVOptionsFrame(BaseFrame):
    """CSV export options frame"""

    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_widgets()

    def create_widgets(self):
        """Create frame components"""
        ttk.Label(self, text="Select Curves for CSV Export").pack(pady=10)

        self.left_var = tk.BooleanVar()
        self.right_var = tk.BooleanVar()
        self.avg_var = tk.BooleanVar()

        ttk.Checkbutton(self, text="Left Ear", variable=self.left_var,
                        command=self.update_state).pack(anchor=tk.W, padx=20)
        ttk.Checkbutton(self, text="Right Ear", variable=self.right_var,
                        command=self.update_state).pack(anchor=tk.W, padx=20)
        ttk.Checkbutton(self, text="Binaural Average", variable=self.avg_var,
                        command=self.update_state).pack(anchor=tk.W, padx=20)

        self.export_btn = ttk.Button(self, text="Export", command=self.process_export,
                                     state=tk.DISABLED)
        self.export_btn.pack(pady=10)
        ttk.Button(self, text="Back", command=self.go_back).pack()

    def update_state(self):
        """Update export button state"""
        if self.left_var.get() or self.right_var.get() or self.avg_var.get():
            self.export_btn.state(['!disabled'])
        else:
            self.export_btn.state(['disabled'])

    def go_back(self):
        """Return to previous frame"""
        if "png" in self.controller.export_formats:
            self.controller.show_frame("ReferenceSelect")
        else:
            self.controller.show_frame("ExportFormat")

    def process_export(self):
        """Prepare CSV selections and start export"""
        self.controller.csv_selections = []
        if self.left_var.get():
            self.controller.csv_selections.append("left")
        if self.right_var.get():
            self.controller.csv_selections.append("right")
        if self.avg_var.get():
            self.controller.csv_selections.append("average")
        self.controller.start_export()

def process_data(controller):
    """Main data processing function"""
    sofa = safe_read_sofa(controller.selected_sofa)
    fs = getattr(sofa, 'Data_SamplingRate')
    irs = getattr(sofa, 'Data_IR')

    output_dir = Path("output") / controller.selected_sofa.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process PNG export
    if "png" in controller.export_formats:
        png_path = output_dir / f"{controller.selected_sofa.stem}.png"
        plot_response(sofa, irs, fs, png_path, controller.reference_file, controller.ref_align_freq)

    # Process CSV export
    if "csv" in controller.export_formats:
        freq, responses = calculate_responses(irs, fs)
        # Apply frequency range filter (to prevent data point at 0hz)
        mask = (freq >= 1)
        filtered_freq = freq[mask]
        for curve in controller.csv_selections:
            idx = ["left", "right", "average"].index(curve)
            filtered_response = responses[idx][mask]
            csv_data = np.column_stack((filtered_freq, filtered_response))
            csv_path = output_dir / f"{controller.selected_sofa.stem}_{curve}.csv"
            np.savetxt(csv_path, csv_data, delimiter=",", fmt='%.6f')


def calculate_responses(irs, fs):
    """Calculate frequency responses for all channels"""
    M, R, N = irs.shape
    freq = np.fft.rfftfreq(N, 1 / fs)

    responses = []
    for r in range(R):
        spectra = np.fft.rfft(irs[:, r, :], axis=1)
        avg_power = np.mean(np.abs(spectra) ** 2, axis=0)
        responses.append(20 * np.log10(np.sqrt(avg_power)))

    if R == 2:
        avg_response = (responses[0] + responses[1]) / 2
        responses.append(avg_response)

    return freq, responses


def safe_read_sofa(file_path):
    """Safely read SOFA file with automatic metadata fixes"""
    try:
        return sofar.read_sofa(file_path, verify=True)
    except ValueError:
        sofa = sofar.read_sofa(file_path, verify=False)
        return sofa


def plot_response(sofa, irs, fs, output_filename, reference_file=None, ref_align_freq=500):
    """Plotting function with reference curve support"""
    freq, responses = calculate_responses(irs, fs)

    # Apply frequency range filter (to prevent data point at 0hz)
    mask = (freq >= 1)
    filtered_freq = freq[mask]
    filtered_responses = [response[mask] for response in responses]

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Color specifications (Linear sRGB colorspace)
    colors = [
        (0.90, 0.10, 0.10),  # Left Ear (Linear sRGB)
        (0.10, 0.90, 0.10),  # Right Ear (Linear sRGB)
        (0.80, 0.03, 0.80)  # Binaural Average (Linear sRGB)
    ]

    # Convert colors to sRGB piecewise for display
    srgb_colors = [linear_to_srgb(c) for c in colors]

    # Plot main responses with absolute SPL levels
    labels = ['Left Ear', 'Right Ear', 'Binaural Average']
    for idx, response in enumerate(filtered_responses):
        plt.semilogx(filtered_freq, response,
                     color=srgb_colors[idx],
                     label=labels[idx],
                     linewidth=2.2,
                     linestyle='-',
                     alpha=1.0)

    # Plot reference curve
    if reference_file and reference_file.exists():
        ref_freq, ref_spl = load_reference_curve(reference_file)
        if ref_freq is not None and len(ref_freq) > 0:
            # Apply frequency range filter (to prevent data point at 0hz)
            ref_mask = (ref_freq >= 1)
            ref_freq = ref_freq[ref_mask]
            ref_spl = ref_spl[ref_mask]

            if len(ref_freq) > 0:
                try:
                    avg_response = filtered_responses[2] if len(filtered_responses) >= 3 else filtered_responses[0]
                    avg_align_idx = np.abs(filtered_freq - ref_align_freq).argmin()
                    ref_align_idx = np.abs(ref_freq - ref_align_freq).argmin()
                    alignment_offset = avg_response[avg_align_idx] - ref_spl[ref_align_idx]
                    ref_normalized = ref_spl + alignment_offset

                    ref_color = linear_to_srgb((0.0, 0.2, 0.4))
                    plt.semilogx(ref_freq, ref_normalized,
                                 color=ref_color,
                                 linewidth=3,
                                 linestyle='--',
                                 alpha=0.7,
                                 label=f'Reference Curve: {reference_file.name}')
                except Exception as e:
                    print(f"Error plotting reference curve: {str(e)}")

    # Configure plot
    ax.set_xlim(10, 30000)
    ax.set_xticks([10, 20, 40, 60, 80, 100, 200, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 20000, 30000])
    ax.set_xticklabels(['10', '20', '40', '60', '80', '100', '200', '500', '1k', '2k', '3k', '4k', '6k', '8k', '10k', '20k', '30k'])
    # ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SPL Level (dB)")
    plt.title(f"Diffuse Field Frequency Response {Path(output_filename).stem}")

    # Force legend update
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        plt.legend(handles, labels, loc='lower center', ncol=2, framealpha=0.9)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()


def linear_to_srgb(color):
    """Convert linear sRGB values to sRGB piecewise transfer function"""
    return tuple(
        c * 12.92 if c <= 0.0031308 else (1.055 * (c ** (1 / 2.4)) - 0.055)
        for c in color
    )


def load_reference_curve(file_path):
    """Load reference curve from text/csv file with robust parsing"""
    try:
        # Detect delimiter and read data with filtering
        with open(file_path, 'r') as f:
            sample = f.read(1024)
            f.seek(0)

            # Detect delimiter and header presence
            try:
                dialect = csv.Sniffer().sniff(sample)
                delimiter = dialect.delimiter
                has_header = csv.Sniffer().has_header(sample)
            except csv.Error:
                delimiter = None
                has_header = False

            # Process lines with header skipping and numeric filtering
            data_lines = []
            for i, line in enumerate(f):
                # Skip initial headers
                if has_header and i == 0:
                    continue

                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Split line using detected delimiter
                parts = line.split(delimiter) if delimiter else line.split()
                if len(parts) != 2:
                    continue

                # Validate numeric values
                try:
                    float(parts[0])
                    float(parts[1])
                    data_lines.append(line)
                except ValueError:
                    continue

            if not data_lines:
                raise ValueError("No valid numeric data found in file")

            # Convert filtered data to numpy array
            data = np.loadtxt(data_lines, delimiter=delimiter)
            return data[:, 0], data[:, 1]

    except Exception as e:
        print(f"Error loading reference curve: {str(e)}")
        return None, None


if __name__ == "__main__":
    Path("input").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    app = AppController()
    app.root.mainloop()