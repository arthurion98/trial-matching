import numpy as np
from pynwb import NWBHDF5IO, NWBFile
from scipy.stats import mode
import os
import pandas as pd

def save_datastructure(load_path="../data/Ghaderi/data_ephys/", save_path="datasets/data_parviz", save_behaviour=True, save_clusters_info=True):
    """ Transform NWB files into a folder structure easily accessible as a pytorch dataset
        Args:
            load_path (str): path to the folder containing the NWB files
            save_path (str): path to the folder where the dataset will be saved
            save_behaviour (bool): whether to save the behavioural data
            save_clusters_info (bool): whether to save the neuronal data
    """

    # create save directory
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # iterate through NWB directory
    nwb_files = [f for f in os.listdir(load_path) if f.endswith('.nwb')]
    for nwb_file in nwb_files:
        nwb_path = os.path.join(load_path, nwb_file)
        save_session(nwb_path, save_path, save_clusters_info)


def save_session(nwb_path, path, save_behaviour=True, save_clusters_info=True):
    """ Save session data into its own directory
        Args:
            nwb_path (str): path to the NWB file of the session
            path (str): path to the folder where the dataset is saved
            save_behaviour (bool): whether to save the behavioural data
            save_clusters_info (bool): whether to save the neuronal data
    """

    with NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()

        trial_df = pd.DataFrame(columns=("trial_number", "audio_cue", "audio_frequency", "whisker_stim", "trial_type", "lick", "early_lick", "trial_onset", "jaw_trace"))
        df_entry = pd.DataFrame(columns=trial_df.columns)

        # creating session directory
        session_id = nwbfile.session_id
        str_splits = session_id.split('-')
        sub_id = str_splits[1].split('_')[0]
        time_id = str_splits[2].split('T')[0]
        session_path = os.path.join(path, sub_id + "_" + time_id)
        if not os.path.exists(session_path):
            os.mkdir(session_path)
        
        # pre-process the jaw coordinates
        jaw_coord = nwbfile.processing['behavior']['BehavioralTimeSeries']['Jaw_Coordinate']
        jaw_coord_x = jaw_coord.data[:, 0]
        jaw_coord_y = jaw_coord.data[:, 1]
        jaw_time = jaw_coord.timestamps[:]

        jaw_coord_x = jaw_coord_x[~np.isnan(jaw_coord_x)]
        jaw_coord_y = jaw_coord_y[~np.isnan(jaw_coord_y)]

        closed_coord = np.array([mode(jaw_coord_x)[0][0], mode(jaw_coord_y)[0][0]])

        # fill the trial dataframe
        trials = nwbfile.trials
        for trial in trials:
            # trial information
            df_entry['trial_number'] = trial.index
            df_entry['audio_cue'] = trial['context'].values[0]
            df_entry['audio_frequency'] = trial['auditory_stim_frequency'].values[0]
            df_entry['whisker_stim'] = trial['whisker_stim'].values[0]
            df_entry['trial_type'] = trial['trial_type'].values[0]
            df_entry['lick'] = trial['lick_flag'].values[0]
            df_entry['early_lick'] = trial['early_lick'].values[0]
            df_entry['trial_onset'] = trial['start_time'].values[0]

            # store behavioural data as npy
            start_idx = np.argmin(np.abs(jaw_time - trial['start_time'].values[0]))
            stop_idx = np.argmin(np.abs(jaw_time - trial['stop_time'].values[0]))

            jaw_trace = np.linalg.norm(jaw_coord.data[start_idx:stop_idx] - closed_coord, axis=1)

            if not os.path.exists(os.path.join(session_path, "jaw_trace")):
                os.mkdir(os.path.join(session_path, "jaw_trace"))
            jaw_path = os.path.join(session_path, "jaw_trace", "trial_{}".format(trial.index[0]))
            if save_behaviour:
                np.save(jaw_path, jaw_trace)
            
            df_entry['jaw_trace'] = jaw_path

            # add entry to trial_df
            trial_df = pd.concat([trial_df, df_entry], ignore_index=True)
        
        trial_df = trial_df.fillna(0)
        trial_df.to_csv(os.path.join(session_path, "trial_info"))
        if save_clusters_info:
            save_clusters(nwbfile, session_path)
    
def save_clusters(nwbfile, session_path):
    """Save the neuronal data into the session directory
        Args:
            nwbfile (NWBFile): NWB file object
            session_path (str): path to the session directory
    """

    cluster_df = pd.DataFrame(columns=("neuron_index", "area", "excitatory", "depth", "cluster", "firing_rate"))
    df_entry = pd.DataFrame(columns=cluster_df.columns)

    for neuron in nwbfile.units:
        if (not neuron['rsUnits'].values[0] and not neuron['fsUnits'].values[0]): # ignore non-excitatory/inhibitory unit
            continue
        excitatory = neuron['rsUnits'].values[0]
        df_entry['neuron_index'] = neuron.index
        df_entry['excitatory'] = [excitatory]
        df_entry['area'] = neuron['location'].values[0]
        df_entry['depth'] = neuron['allenccf_area_layer'].values[0]
        df_entry['cluster'] = neuron['cluster_id'].values[0]
        df_entry['firing_rate'] = neuron['firing_rate'].values[0]

        spikes = neuron['spike_times'].values[:][0]
        np.save(os.path.join(session_path, "neuron_index_{}".format(neuron.index[0])), spikes)

        cluster_df = pd.concat([cluster_df, df_entry], ignore_index=True)
    cluster_df.to_csv(os.path.join(session_path, "cluster_info"))

def unified_cluster_table(path="datasets/data_parviz"):
    """Consolidate the cluster information of all sessions into a single csv file.
        Args:
            path (str, optional): path to the folder containing the session directories.
    """

    cluster_information = pd.DataFrame(columns=("session", "area", "excitatory", "firing_rate", "cluster_index"))
    dirs = os.listdir(path)
    if "cluster_information" in dirs:
        dirs.remove("cluster_information")
    df_entry = {}
    for dir in dirs:
        try:
            df = pd.read_csv(os.path.join(path, dir, "cluster_info"))
        except:
            pass
        df_entry = pd.DataFrame(columns=cluster_information.columns)
        df_entry["session"] = [dir]
        df.neuron_index = df.index.values
        df.to_csv(os.path.join(path, dir, "cluster_info"))
        for index, row in df.iterrows():
            df_entry["area"] = [row.area]
            df_entry["excitatory"] = [row.excitatory]
            df_entry["firing_rate"] = [row.firing_rate]
            df_entry["cluster_index"] = [index]
            cluster_information = pd.concat([cluster_information, df_entry], ignore_index=True)
    cluster_information.to_csv(os.path.join(path, "cluster_information"))

if __name__ == "__main__":
    save_datastructure()
    unified_cluster_table()