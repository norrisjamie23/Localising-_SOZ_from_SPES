import mne
import pandas as pd
from mne_bids import BIDSPath, get_entity_vals
import numpy as np
import pyreadr

def pad_and_stack(arrays, max_rows, pad_value=0):
    """
    Pad and stack arrays.

    Args:
        arrays (list): List of arrays to be padded and stacked.
        max_rows (int): Maximum number of rows for padding.
        pad_value (int, optional): Value used for padding. Defaults to 0.

    Returns:
        numpy.ndarray: Stacked array with padded arrays.

    """
    # Pad each array as necessary and collect them in a new list
    padded_arrays = []
    for array in arrays:
        rows_to_add = max_rows - array.shape[0]
        if rows_to_add > 0:
            # Pad below the array
            array = np.pad(array, ((0, rows_to_add), (0, 0)), 'constant', constant_values=(pad_value,))
        padded_arrays.append(array)
    
    # Stack the padded arrays
    stacked_array = np.stack(padded_arrays)
    return stacked_array

# Load the .rda file
destrieux_df = pyreadr.read_r('../destrieux.rda')['destrieux']

def get_destrieux_lobe(label):
    """
    Returns the lobe name corresponding to the given label.

    Parameters:
    label (int): The label value.

    Returns:
    str: The name of the lobe corresponding to the label.
    """
    if label == 0:
        return 'Unknown'
    else:
        return destrieux_df[destrieux_df.index == label - 1].lobe.values[0]

class BIDSDataLoader:
    """
    A class for loading data from the Brain Imaging Data Structure (BIDS) format.

    Parameters:
    - bids_root (str): The root directory of the BIDS dataset.

    Methods:
    - load_subjects(): Load the list of subjects in the dataset.
    - load_session_data(subject): Load session data for a given subject.
    - load_run_data(subject, run): Load data for a specific run within a subject's session.
    """

    def __init__(self, bids_root):
        self.bids_root = bids_root
        self.bids_path = BIDSPath(root=bids_root)

    def load_subjects(self):
        subjects = get_entity_vals(self.bids_root, 'subject')
        return subjects

    def load_session_data(self, subject):
        """
        Load session data for a given subject.
        Returns a dictionary with electrodes info and runs.

        Parameters:
        - subject (str): The subject ID.

        Returns:
        - dict: A dictionary containing the following keys:
            - "electrodes_tsv" (pd.DataFrame): The electrodes information as a pandas DataFrame.
            - "runs" (list): The list of runs for the subject.
        """
        # Update the path for the current subject and specific task
        patient_bids_path = self.bids_path.copy().update(subject=subject, datatype="ieeg", task="SPESclin")
        
        # Find all .vhdr files for the subject and task
        vhdr_paths = patient_bids_path.copy().update(extension=".vhdr").match()
        
        # Only if it ends in .vhdr
        vhdr_paths = [vhdr_path for vhdr_path in vhdr_paths if str(vhdr_path.fpath)[-5:] == ".vhdr"]

        # Find all .vhdr files for the subject and task
        if not vhdr_paths:
            raise ValueError(f"No .vhdr files found for subject {subject}")

        # Extract the sessionf from the first .vhdr file's path
        session = vhdr_paths[0]._session

        # Get the path to the electrodes file for the current session
        electrode_path = patient_bids_path.copy().update(task=None, session=session, extension='.tsv', suffix='electrodes').match()[0].fpath
        
        # Load electrodes file as pandas DataFrame
        electrodes_tsv = pd.read_csv(electrode_path, sep="\t", index_col=0)

        # List all runs by extracting the run number from each .vhdr file's path
        runs = [vhdr_path._run for vhdr_path in vhdr_paths]

        return {
            "electrodes_tsv": electrodes_tsv,
            "runs": runs
        }

    def load_run_data(self, subject, run):
        """
        Load data for a specific run within a subject's session.
        Returns a dictionary with EEG data, events data, and channels data.

        Parameters:
        - subject (str): The subject ID.
        - run (str): The run number.

        Returns:
        - dict: A dictionary containing the following keys:
            - "eeg" (mne.io.Raw): The EEG data.
            - "events_df" (pd.DataFrame): The events data as a pandas DataFrame.
            - "channels_df" (pd.DataFrame): The channels data as a pandas DataFrame.
        """
        # Update the path for the current subject, task, and run
        patient_bids_path = self.bids_path.copy().update(subject=subject, datatype="ieeg", task="SPESclin", run=run)

        # Find all .vhdr files for the subject and task
        vhdr_paths = patient_bids_path.copy().update(extension=".vhdr").match()
        
        # Only if it ends in .vhdr
        vhdr_paths = [vhdr_path for vhdr_path in vhdr_paths if str(vhdr_path.fpath)[-5:] == ".vhdr"]

        # Get the path to the .vhdr file for the current run
        vhdr_path = vhdr_paths[0].fpath
        
        # Get the path to the events file for the current run
        tsv_path = patient_bids_path.copy().update(extension=".tsv", suffix="events").match()[0].fpath
        
        # Get the path to the channels file for the current run
        channels_path = patient_bids_path.copy().update(extension=".tsv", suffix="channels").match()[0].fpath

        eeg = mne.io.read_raw_brainvision(vhdr_path, verbose=False, preload=True) # TODO
        events_df = pd.read_csv(tsv_path, sep="\t", index_col=0)
        channels_df = pd.read_csv(channels_path, sep="\t", index_col=0)

        return {
            "eeg": eeg,
            "events_df": events_df,
            "channels_df": channels_df
        }

# Function to split, sort, and recombine the values
def process_stimulation_sites(site):
    """
    Process the stimulation site string by splitting it into parts, sorting the parts, and joining them back together.

    Args:
        site (str): The stimulation site string to be processed.

    Returns:
        str: The processed stimulation site string.

    """
    parts = site.split('-')  # Split the string into parts
    parts.sort()             # Sort the parts
    return '-'.join(parts)

def is_overlap(event, artefact):
    return (event['sample_start'] < artefact['sample_end']) and (artefact['sample_start'] < event['sample_end'])

# Function to combine mean and standard deviation
def combine_stats(group):
    """
    Combine statistics for a group of data.

    Parameters:
    - group: pandas DataFrame
        The group of data to combine statistics for.

    Returns:
    - mean_df: pandas DataFrame
        DataFrame containing the combined mean values.
    - std_df: pandas DataFrame
        DataFrame containing the combined standard deviation values.
    """
    # Filter rows and keep only numeric columns
    means = group[group['metric'] == 'mean'].select_dtypes(include=[np.number])
    stds = group[group['metric'] == 'std'].select_dtypes(include=[np.number])

    # Compute combined mean and standard deviation
    n = 10  # number of trials each row represents
    if len(means) > 1 or len(stds) > 1:
        total_samples = n * len(means)
        combined_mean = means.mean()
        combined_std = np.sqrt((stds**2).mean() + ((means - combined_mean)**2).sum() / (total_samples - 1))
    else:
        combined_mean = means.iloc[0]
        combined_std = stds.iloc[0]

    mean_df = pd.DataFrame(combined_mean).transpose()
    std_df = pd.DataFrame(combined_std).transpose()

    for col in ['subject', 'recording', 'stim_1', 'stim_2']:
        mean_df[col] = group[col].iloc[0]
        std_df[col] = group[col].iloc[0]

    mean_df['metric'] = 'mean'
    std_df['metric'] = 'std'

    return mean_df, std_df

regions = ['Frontal', 'Insula', 'Limbic', 'Occipital', 'Parietal', 'Temporal', 'Unknown']
region_to_index = {region: index for index, region in enumerate(regions)}

class StimulationDataProcessor:
    def __init__(self, tmin, tmax):
        """
        Initialize the class with minimum and maximum temperature values.

        Args:
            tmin (float): The minimum temperature value.
            tmax (float): The maximum temperature value.
        """
        self.tmin = tmin
        self.tmax = tmax

    def process_run_data(self, eeg, events_df, channels_df, subject):
        """
        Process and extract stimulation data for a given run.
        Returns a list of dictionaries, each representing a processed epoch.
        """

        # Filter channels
        chans_to_use = channels_df[channels_df.status_description == "included"].index.tolist()
        eeg.pick(chans_to_use)

        # Filter EEG data
        eeg.filter(1, 150, n_jobs=-1, method='fir', fir_design='firwin')  # TODO

        # Filter for electrical stimulation events
        stim_events = events_df[events_df.trial_type == "electrical_stimulation"].copy()
        
        # Filter for events that occur within the EEG data
        before = stim_events.shape[0]
        stim_events = stim_events[stim_events['sample_start'] < eeg.tmax * eeg.info['sfreq']]
        after = stim_events.shape[0]

        if before != after:
            print(subject, before, after)

        # Filter for artefact events
        artefacts = events_df[np.logical_and(events_df.trial_type == "artefact", events_df.electrodes_involved_onset == "all")].copy()

        # Filter for seizure events
        seizures = events_df[events_df.trial_type == "seizure"].copy()

        # Creating artefact mask
        artefact_mask = []
        for _, stim_event in stim_events.iterrows():
            overlap = any(is_overlap(stim_event, artefact) for _, artefact in artefacts.iterrows())
            artefact_mask.append(overlap)

        # Creating the artmask
        seizure_mask = []
        for _, stim_event in stim_events.iterrows():
            overlap = any(is_overlap(stim_event, seizure) for _, seizure in seizures.iterrows())
            seizure_mask.append(overlap)

        # Create mask of valid events
        valid_mask = np.logical_not(np.logical_or(artefact_mask, seizure_mask))

        # Filter for valid events
        stim_events = stim_events[valid_mask]

        # Filter for artefact events occurring on only some channels
        focal_artefacts = events_df[np.logical_and(events_df.trial_type == "artefact", events_df.electrodes_involved_onset != "all")].copy()

        # Sort the 'electrical_stimulation_site' column - this ensures that E2-E1 and E1-E2 are treated the same
        stim_events['electrical_stimulation_site'] = stim_events['electrical_stimulation_site'].apply(process_stimulation_sites)

        # Step 1: Convert 'electrical_stimulation_site' to a categorical datatype
        stim_events['electrical_stimulation_site'] = stim_events['electrical_stimulation_site'].astype('category')

        # Get mapping of categories to integer codes for 'electrical_stimulation_site'
        category_mapping = dict(enumerate(stim_events['electrical_stimulation_site'].cat.categories))

        # Create a new column 'electrical_stimulation_site_cat' with integer codes for 'electrical_stimulation_site'
        stim_events['electrical_stimulation_site_cat'] = stim_events['electrical_stimulation_site'].cat.codes

        # Step 2: Add a column 'zero_column' to the DataFrame with all values set to 0
        stim_events['zero_column'] = 0

        # Step 3: Extract the columns 'sample_start', 'zero_column', and 'electrical_stimulation_site_cat' and convert to a NumPy array
        result_array = stim_events[['sample_start', 'zero_column', 'electrical_stimulation_site_cat']].values

        # Creating the artmask
        overlap_list = []

        for _, stim_event in stim_events.iterrows():
            overlap_found = False
            for _, artefact in focal_artefacts.iterrows():
                if is_overlap(stim_event, artefact):
                    overlap_list.append(artefact.electrodes_involved_onset)
                    overlap_found = True
                    break  # Stop checking after the first overlap is found
            if not overlap_found:
                overlap_list.append(None)
        overlap_list = np.array(overlap_list)

        response_dfs = []
        
        for event_id in np.unique(result_array[:, 2]):

            # Chans to remove due to artefact
            remove_chans = overlap_list[result_array[:, 2] == event_id]
            remove_chans = np.unique([chan for chan_list in remove_chans if chan_list is not None for chan in chan_list.split(',')])

            response_df = self._extract_epochs(eeg, result_array, event_id, category_mapping, subject, remove_chans)
            response_dfs.append(response_df)

        try:
            response_dfs = pd.concat(response_dfs)
        except ValueError:
            print(f"No stimulation events found for subject {subject}")
            return None        

        return response_dfs

    def _extract_epochs(self, eeg, result_array, event_id, category_mapping, subject, remove_chans):
        """
        Extracts an epoch for a given stimulation event.
        """
        stimulated_electrodes = category_mapping[event_id].split('-')

        # When polarity is reversed, ensure no duplication
        stimulated_electrodes.sort()

        # Get list of electrodes excluding stimulated electrodes
        recording_channels = [chan for chan in eeg.info['ch_names'] if chan not in stimulated_electrodes]

        # Remove channels that are artefacted
        recording_channels = [chan for chan in recording_channels if chan not in remove_chans]

        epochs = mne.Epochs(eeg, result_array, event_id=event_id, tmin=self.tmin - 1, tmax=self.tmax, picks=recording_channels, preload=True, baseline=(None, -0.1))
        epochs.crop(tmin=self.tmin)
        epochs.resample(512)

        # If less than 5 trials, return None
        if (epochs._data.shape[0]) < 5:
            return None
                
        # Calculate mean and standard deviation
        mean_response = epochs.average()._data  # Shape: (channels, time steps)
        std_response = epochs._data.std(axis=0)  # Shape: (channels, time steps)

        # Create DataFrame for mean response
        df_mean_response = pd.DataFrame(mean_response).astype('float32')
        df_mean_response.insert(0, 'subject', [subject] * mean_response.shape[0])
        df_mean_response.insert(1, 'recording', epochs.info['ch_names'])
        df_mean_response.insert(2, 'stim_1', stimulated_electrodes[0])
        df_mean_response.insert(3, 'stim_2', stimulated_electrodes[1])
        df_mean_response.insert(4, 'metric', 'mean')

        # Create DataFrame for std response
        df_std_response = pd.DataFrame(std_response).astype('float32')
        df_std_response.insert(0, 'subject', [subject] * std_response.shape[0])
        df_std_response.insert(1, 'recording', epochs.info['ch_names'])
        df_std_response.insert(2, 'stim_1', stimulated_electrodes[0])
        df_std_response.insert(3, 'stim_2', stimulated_electrodes[1])
        df_std_response.insert(4, 'metric', 'std')

        # Concatenate the two DataFrames
        response_df = pd.concat([df_mean_response, df_std_response], ignore_index=True)

        return response_df

class DatasetCreator:
    """
    A class for creating datasets for analysis.

    Args:
        response_df (pandas.DataFrame): The response dataframe containing subject data.

    Methods:
        process_for_analysis: Process the data for analysis.

    """

    def __init__(self, response_df):
        self.response_df = response_df

    def process_for_analysis(self, subject, electrodes_df):
        """
        Process the stimulation data for analysis.

        Args:
            subject (str): The subject identifier.
            electrodes_df (pandas.DataFrame): The dataframe containing the stimulation data.

        Returns:
            None
        """
        self.process_metric_for_analysis(subject, electrodes_df, 'mean', labels=True)
        self.process_metric_for_analysis(subject, electrodes_df, 'std')

    def process_metric_for_analysis(self, subject, electrodes_df, metric, labels=False):
        """
        Process the data for analysis.

        Args:
            subject (str): The subject identifier.
            electrodes_df (pandas.DataFrame): The dataframe containing electrode data.
            metric (str): The metric to be analysed.
            labels (bool, optional): Whether to include labels. Defaults to False.

        Returns:
            numpy.ndarray or None: The processed data for analysis or None if no stimulation events found.

        """
        response_df = self.response_df[self.response_df.subject == subject]
        response_df = response_df[response_df.metric == metric]

        try:
            # Calculate stimulation and recording coordinates
            stim_1_coords = np.array([[electrodes_df[electrodes_df.index == stimulated_electrode].x, electrodes_df[electrodes_df.index == stimulated_electrode].y, electrodes_df[electrodes_df.index == stimulated_electrode].z] for stimulated_electrode in response_df.stim_1])
            stim_2_coords = np.array([[electrodes_df[electrodes_df.index == stimulated_electrode].x, electrodes_df[electrodes_df.index == stimulated_electrode].y, electrodes_df[electrodes_df.index == stimulated_electrode].z] for stimulated_electrode in response_df.stim_2])
            stim_coords = (stim_1_coords + stim_2_coords) / 2

            recording_coords = np.array([[electrodes_df[electrodes_df.index == stimulated_electrode].x, electrodes_df[electrodes_df.index == stimulated_electrode].y, electrodes_df[electrodes_df.index == stimulated_electrode].z] for stimulated_electrode in response_df.recording])

            # Calculate Euclidean distance
            distances = np.sqrt(np.sum((stim_coords - recording_coords) ** 2, axis=1))[:, 0]

            # Keep only distances greater than 13mm
            response_df = response_df[distances > 13]
            distances = distances[distances > 13]

            # Add distances to DataFrame
            response_df['distances'] = distances

            # Sort by distances - used for CNN method
            response_df = response_df.sort_values(by='distances', ascending=True)

        except:
            print("Error with subject", subject)

        # Get channels used for both stimulation and recording
        recording_stim_channels = set(response_df.recording.unique()).intersection(set(response_df.stim_2.unique()).union(set(response_df.stim_1.unique())))

        channels_recording_trials, channels_stim_trials = [], []
        
        if labels:
            channel_soz = []

        if len(recording_stim_channels) == 0:
            print(f"No stimulation events found for subject {subject}")
            return None
        
        electrode_coords = []
        electrode_lobes = []

        for channel in recording_stim_channels:

            # Current channel responses when other channels were stimulated
            channel_recording_trials = response_df[response_df.recording == channel].select_dtypes(include='number').copy()
            
            # Other channel responses when current channel was stimulated
            channel_stim_trials = response_df[np.logical_or(response_df.stim_1 == channel, response_df.stim_2 == channel)].select_dtypes(include='number').copy()

            # Add to corresponding lists, except for recording/stim channel names (i.e., only time series)
            channels_recording_trials.append(np.array(channel_recording_trials))
            channels_stim_trials.append(np.array(channel_stim_trials))

            if labels:
                # Add label for current channel
                channel_soz.append(electrodes_df[electrodes_df.index == channel].soz.iloc[0] == "yes")

            electrode_coords.append([electrodes_df[electrodes_df.index == channel].x.iloc[0], electrodes_df[electrodes_df.index == channel].y.iloc[0], electrodes_df[electrodes_df.index == channel].z.iloc[0]])
        
            electrode_lobe = electrodes_df[electrodes_df.index == channel].Destrieux_label.values[0]
            electrode_lobe = get_destrieux_lobe(electrode_lobe)
            electrode_lobe = region_to_index[electrode_lobe]
            electrode_lobes.append(electrode_lobe)
        
        # For recording channels
        max_recording_rows = max(array.shape[0] for array in channels_recording_trials)
        X_recording = pad_and_stack(channels_recording_trials, max_recording_rows).astype(np.float32)

        # For stim channels
        max_stim_rows = max(array.shape[0] for array in channels_stim_trials)
        X_stim = pad_and_stack(channels_stim_trials, max_stim_rows).astype(np.float32)

        if labels:
            y = np.array(channel_soz, dtype=np.int32)
            if y.sum() == 0:
                return None
            np.save(f'../data/main/lobes_{subject}.npy', np.array(electrode_lobes))
            np.save(f'../data/main/y_{subject}.npy', y)
            np.save(f'../data/main/coords_{subject}.npy', np.array(electrode_coords))

        # Save as np arrays
        np.save(f'../data/{metric}/X_stim_{subject}.npy', X_stim)
        np.save(f'../data/{metric}/X_recording_{subject}.npy', X_recording)