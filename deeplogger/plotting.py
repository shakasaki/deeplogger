import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_atv_am(data, figure_limits: list = [0, 359, 250, 255], cmap: str = 'hot', clims: list = None, title: str = None):
    '''Plot the data
    Args:
        data: data to plot
        figure_limits: limits of the figure
        cmap: colormap
        clims: color limits
        title: title of the figure
    Returns:
        fig, axs: figure and axis of the plot
    '''
    fig, axs = plt.subplots()
    im = axs.imshow(data, aspect='auto', extent=figure_limits, cmap=cmap)
    if clims is not None:
        im.set_clim(min(clims), max(clims))
    axs.set_xlabel('Azimuth [deg]')
    axs.set_ylabel('Depth [m]')
    if title is not None:
        fig.suptitle(title)

    # Create a colorbar based on the mappable 'im'
    cbar = plt.colorbar(im)
    cbar.set_label('Amplitude [-]')

    plt.show()
    return fig, axs

def plot_atv_tt(data, figure_limits: list = [0, 359, 250, 255], cmap: str = 'hot', clims: list = None, title: str = None):
    '''Plot the comparison before and after processing
    Args:
        data: data to plot
        figure_limits: limits of the figure
        cmap: colormap
        clims: color limits
        title: title of the figure
    Returns:
        fig, axs: figure and axis of the plot
    '''
    fig, axs = plt.subplots()
    im = axs.imshow(data, aspect='auto', extent=figure_limits, cmap=cmap)
    if clims is not None:
        im.set_clim(min(clims), max(clims))
    axs.set_xlabel('Azimuth [deg]')
    axs.set_ylabel('Depth [m]')
    if title is not None:
        fig.suptitle(title)

    # Create a colorbar based on the mappable 'im'
    cbar = plt.colorbar(im)
    cbar.set_label('Travel Time [us]')

    plt.show()
    return fig, axs



def plot_comparison_am(data_start, data_processed, removed_data, figure_limits: list = [0, 359, 250, 255],
                       subplot_widths: list = [1.35, 1.05, 1],  # Adjust the widths of the subplots
                       cmap: str = 'hot', clims: list = None, title: str = None):
    '''Plot the comparison before and after processing
    Args:
        data_start: initial data
        data_processed: data after processing
        removed_data: data that is removed by the processing
        figure_limits: limits of the figure
        subplot_widths: a list of relative widths for each subplot
        cmap: colormap
        clims: color limits
        title: title of the figure
    Returns:
        fig, axs: figure and axis of the plot
    '''
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=subplot_widths)

    axs = gs.subplots()

    for data, ax in zip([data_start, data_processed, removed_data], axs):
        im = ax.imshow(data, aspect='auto', extent=figure_limits, cmap=cmap)
        if clims is not None:
            im.set_clim(min(clims), max(clims))
        ax.set_xlabel('Azimuth [deg]')
        ax.set_ylabel('Depth [m]')

        if ax in [axs[0], axs[1]]:
            # Remove tick labels and the y-label for "Processed data" and "Removed data"
            ax.set_yticks([])
            ax.set_ylabel('')
        if ax in [axs[2]]:
            # Remove tick labels and the y-label for "Processed data" and "Removed data"
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

    axs[0].set_title('Initial data')
    axs[1].set_title('Processed data')
    axs[2].set_title('Removed data')

    # Create colorbars next to the plots that use colormaps
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("left", size="5%", pad=0.80)
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Amplitude [-]')  # Move the title to the left

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    return fig, axs



def plot_comparison_tt(data_start, data_processed, removed_data, figure_limits: list = [0, 359, 250, 255],
                       subplot_widths: list = [1.35, 1.05, 1],  # Adjust the widths of the subplots
                       cmap: str = 'hot', clims: list = None, title: str = None):
    '''Plot the comparison before and after processing
    Args:
        data_start: initial data
        data_processed: data after processing
        removed_data: data that is removed by the processing
        figure_limits: limits of the figure
        subplot_widths: a list of relative widths for each subplot
        cmap: colormap
        clims: color limits
        title: title of the figure
    Returns:
        fig, axs: figure and axis of the plot
    '''
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=subplot_widths)

    axs = gs.subplots()

    for data, ax in zip([data_start, data_processed, removed_data], axs):
        im = ax.imshow(data, aspect='auto', extent=figure_limits, cmap=cmap)
        if clims is not None:
            im.set_clim(min(clims), max(clims))
        ax.set_xlabel('Azimuth [deg]')
        ax.set_ylabel('Depth [m]')

        if ax in [axs[0], axs[1]]:
            # Remove tick labels and the y-label for "Processed data" and "Removed data"
            ax.set_yticks([])
            ax.set_ylabel('')
        if ax in [axs[2]]:
            # Remove tick labels and the y-label for "Processed data" and "Removed data"
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

    axs[0].set_title('Initial data')
    axs[1].set_title('Processed data')
    axs[2].set_title('Removed data')

    # Create colorbars next to the plots that use colormaps
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("left", size="5%", pad=0.80)
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Travel Time [us]')  # Move the title to the left

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    return fig, axs


# Example usage:
# plot_comparison_am(data_start, data_processed, removed_data, title="Data Comparison")

