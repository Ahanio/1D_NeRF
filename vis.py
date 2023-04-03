import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider, Button, RadioButtons
from data import shoot_one_ray


def to_np(x):
    return x.detach().cpu().numpy()


class Visualizer:
    def __init__(self, x_range, objects, model, enable=True):
        self.x_range = x_range
        self.objects = objects
        self.lines = []
        self.enable = enable
        self.model = model

        if self.enable:
            plt.ion()
            self.init_plot()

    def init_plot(
        self,
    ):
        self.fig, self.ax = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

        self.ax[0].set_title("Density")
        self.ax[0].set_xlim(self.x_range)

        self.ax[1].set_title("Transmittance")
        self.ax[1].set_ylim(0, 1.2)

        self.ax[2].set_title("Surface")
        self.ax[2].set_ylim(0, 0.1)

        for num_plot in range(3):
            self.ax[num_plot].axvspan(
                -100, self.x_range[0] + 0.05, color="red", alpha=0.1
            )
            self.ax[num_plot].axvspan(
                self.x_range[1] - 0.05, 100, color="red", alpha=0.1
            )
            for obj in self.objects:
                self.ax[num_plot].axvspan(obj[0], obj[1], color="red", alpha=0.1)

            # [a.legend() for a in self.ax]

    def plot(self, x, density, tr, w, color="blue"):
        if self.enable:
            # plot params with smaller points size
            plot_params = {"linewidth": 1, "c": color, "markersize": 4}
            for num_ax, to_plot in enumerate([to_np(density), to_np(tr), to_np(w)]):
                if num_ax == 0:
                    x_density = torch.linspace(self.x_range[0], self.x_range[1], 300)[
                        :, None
                    ]
                    full_density = self.model(x_density)
                    a = self.ax[num_ax].plot(
                        to_np(x_density), to_np(full_density), "-o", **plot_params
                    )
                elif num_ax == 1:
                    a = self.ax[num_ax].plot(to_np(x), to_plot, "-o", **plot_params)
                elif num_ax == 2:
                    a = self.ax[num_ax].plot(to_np(x), to_plot, "-o", **plot_params)
                    self.ax[2].set_ylim(0, max(0.1, to_plot.max()))
                self.lines.append(a)

    def clear_plots(self):
        if self.enable:
            for cur_ax in self.ax:
                for line in cur_ax.get_lines():
                    line.remove()

    def show(self, block=False, interactive=False):
        if self.enable:
            if interactive:
                self.interactive_mode()
            plt.ion()
            plt.show(block=block)
            plt.pause(0.05)

    def save(self, path):
        if self.enable:
            self.fig.savefig(path)

    def close(self):
        if self.enable:
            plt.close(self.fig)

    def interactive_mode(self):
        "Run interactive plot with sliders. One slider specifies the position of the camera, other - direction of ray."
        if not self.enable:
            return

        axfreq = self.fig.add_axes([0.15, 0.01, 0.75, 0.03])
        self.freq_slider = Slider(
            ax=axfreq,
            label="Position",
            valmin=self.x_range[0],
            valmax=self.x_range[1],
            valinit=0.1,
        )

        rax = self.fig.add_axes([0.01, 0.03, 0.05, 0.1])
        self.radio = RadioButtons(
            rax,
            (
                "1",
                "-1",
            ),
        )

        self.freq_slider.on_changed(self.update)
        self.radio.on_clicked(self.update)

    def update(self, val):
        with torch.no_grad():
            self.clear_plots()

            origin = torch.tensor(
                [
                    self.freq_slider.val,
                ]
            ).float()
            direction = torch.tensor(
                [
                    float(self.radio.value_selected),
                ]
            ).float()
            x, delta = shoot_one_ray(origin, direction, 300, self.x_range)

            density = self.model(x.reshape(-1, 1))
            tr = self.model.transmittance(density, delta)
            w = self.model.surface(tr, density, delta)

            self.plot(x, density, tr, w)
            # self.fig.canvas.draw_idle()
