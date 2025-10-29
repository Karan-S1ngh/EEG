import mne
import matplotlib.pyplot as plt
import os
import warnings

ASSUMED_CHANNEL_NAMES = ['Fz', 'Cz', 'Pz', 'Oz', 'C3', 'C4']
CHANNEL_MAP = {f'Ch{i+1}': name for i, name in enumerate(ASSUMED_CHANNEL_NAMES)}
PLOT_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'saved_plots'))
PLOT_NAME = 'channel_locations.png'

warnings.filterwarnings('ignore') # Suppress MNE info messages

def plot_channel_locations():
    """
    Creates and plots an assumed 6-channel EEG layout using MNE.
    V2: Uses sphere='auto' for plotting.
    """
    print("Assuming 6 channels correspond to:", ASSUMED_CHANNEL_NAMES)

    # 1. Create MNE Info object
    info = mne.create_info(ch_names=ASSUMED_CHANNEL_NAMES, sfreq=200, ch_types='eeg')

    # 2. Set the standard montage
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage, on_missing='ignore')
    except Exception as e:
        print(f"Error setting montage: {e}. Cannot plot locations.")
        return

    # 3. Plot the sensor locations
    print("Generating plot of assumed channel locations...")
    try:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        mne.viz.plot_sensors(info, kind='topomap', sphere='auto', show_names=True, axes=ax, show=False)
        ax.set_title("Assumed EEG Channel Locations (Standard 10-20 System)")
        plt.tight_layout()

        # Save the plot
        if not os.path.exists(PLOT_SAVE_PATH): os.makedirs(PLOT_SAVE_PATH)
        plot_filepath = os.path.join(PLOT_SAVE_PATH, PLOT_NAME)
        plt.savefig(plot_filepath)
        plt.close(fig)
        print(f"\nSuccessfully saved channel location plot to '{plot_filepath}'")
        print("\nChannel Mapping Used:")
        for ch_num, ch_name in CHANNEL_MAP.items():
            print(f"  {ch_num} -> {ch_name}")

    except Exception as e:
        print(f"Error generating plot: {e}")
        

if __name__ == '__main__':
    plot_channel_locations()