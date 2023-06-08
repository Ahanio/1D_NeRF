import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, RadioButtons, Slider

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
        self.geom_type = model.geom_type

        if self.enable:
            plt.ion()
            self.init_plot()

    def init_plot(
        self,
    ):
        self.fig, self.ax = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        plt.setp(
            self.ax,
            xticks=np.arange(*self.x_range, step=1),
        )

        self.ax[0].set_title(self.geom_type)
        self.ax[0].set_xlim(self.x_range)
        self.ax[0].set_ylim(0, 1.2)

        self.ax[1].set_title("Transmittance")
        self.ax[1].set_ylim(0, 1.2)

        self.ax[2].set_title("PDF")
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

    def plot(self, x, density, tr, w, color="blue", full_range_density=True):
        # full_range_density – whether to plot density for the whole range or only for the specified ray
        if self.enable:
            # plot params with smaller points size
            plot_params = {"linewidth": 1, "c": color, "markersize": 4}

            # ax[0] – density/occupancy
            # ax[1] – Transmittance
            # ax[2] – Surface

            for num_ax, to_plot in enumerate([to_np(density), to_np(tr), to_np(w)]):
                if num_ax == 0:
                    if self.model.name == "NeRF":
                        x_density, _ = shoot_one_ray(0, 1, 300, self.x_range)

                    elif self.model.name == "Unisurf":
                        num_samples = int(
                            (self.x_range[1] - self.x_range[0]) / self.model.step_size
                        )
                        x_density = torch.arange(
                            0, self.model.step_size * num_samples, self.model.step_size
                        )[:, None]

                    full_geom = self.model(x_density) if full_range_density else density

                    a = self.ax[num_ax].plot(
                        to_np(x_density), to_np(full_geom), "-o", **plot_params
                    )
                    if self.geom_type == "density":
                        self.ax[0].set_ylim(0, max(5, max(to_np(full_geom)) + 0.5))

                elif num_ax == 1:
                    a = self.ax[num_ax].plot(to_np(x), to_plot, "-o", **plot_params)
                    # a = self.ax[num_ax].plot(to_np(x_density), to_np(full_density), "-o", **plot_params, alpha=0.1)

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

    def interactive_mode(self):
        """Run interactive plot with sliders.
        One slider specifies the position of the camera,
        other - direction of ray."""
        if not self.enable:
            return

        axfreq = self.fig.add_axes([0.12, 0.01, 0.78, 0.03])
        self.freq_slider = Slider(
            ax=axfreq,
            label="Position",
            valmin=self.x_range[0],
            valmax=self.x_range[1],
            valinit=0.1,
        )

        rax = self.fig.add_axes([0.01, 0.04, 0.04, 0.1])
        self.radio = RadioButtons(
            rax,
            (
                "–>",
                "<–",
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
            direction = 1 if self.radio.value_selected == "–>" else -1
            direction = torch.tensor(
                [
                    float(direction),
                ]
            ).float()
            x, delta = shoot_one_ray(origin, direction, 300, self.x_range)

            geom = self.model(x.reshape(-1, 1)).view(1, -1)
            tr = self.model.transmittance(geom, delta)
            w = self.model.surface(tr, geom, delta)
            local_x = torch.abs(x - origin)
            pred = torch.sum(w * local_x, dim=1)

            print("Predicted depth:", pred)
            print("W sum:", w.sum().item())
            self.plot(x, geom[0], tr[0], w[0])

    def save(self, path):
        if self.enable:
            self.fig.savefig(path)

    def close(self):
        if self.enable:
            plt.close(self.fig)
