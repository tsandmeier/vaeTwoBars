import os

from dataset.helpers import *

import numpy as np
from dataset.constants_latent_factors import INTERVAL_DICT, RHYTHMIC_ENTROPY_DICT, UNIQUE_PITCHES_DICT


class DoodleDataset:
    def __init__(self):
        self.dataset_path = os.path.join("data",
                                         "doodle_dataset_downloaded")
        self.score_array = None
        self.latent_array = None
        self.metadata = None

        self.note2index_dict = dict()
        self.index2note_dict = dict()
        self.initialize_index_dicts()
        self.beat_subdivisions = len(TICK_VALUES)

        self.tick_durations = compute_tick_durations(TICK_VALUES)

        self.latent_dicts = {
            'rhythmic_entropy': RHYTHMIC_ENTROPY_DICT,
            'interval': INTERVAL_DICT,
            'unique_pitches': UNIQUE_PITCHES_DICT
        }

    def initialize_index_dicts(self):
        """
        Reads index dicts from file if available, else creates it

        """
        note_sets = set()
        # add rest and continue symbols
        note_sets.add('rest')
        note_sets.add(CONTINUE_SYMBOL)
        for note_index, note_name in enumerate(note_sets):
            self.index2note_dict.update({note_index: note_name})
            self.note2index_dict.update({note_name: note_index})

        # self.beat_subdivisions = len(TICK_VALUES)
        # self.tick_durations = compute_tick_durations(TICK_VALUES)

        # self.counter = 0

    def make_or_load_dataset(self):
        """
            Creates the dataset or reads if it already exists
            Returns:
                None
            """

        if os.path.exists(self.dataset_path + '.npz'):
            print('Dataset already created. Reading it now')
            dataset = np.load(self.dataset_path + '.npz', allow_pickle=True)
            self.score_array = dataset['score_array']
            self.latent_array = dataset['latent_array']
            self.note2index_dict = dataset['note2index_dict'].item()
            self.index2note_dict = dataset['index2note_dict'].item()
            self.latent_dicts = dataset['latent_dicts'].item()
            # self.metadata = dataset['metadata'].item()
            return

        print('Making tensor dataset')
        score_seq = [None] * 1000000
        latent_seq = [None] * 1000000

        def _create_data_point(item_index, notelist):

            if notelist['input_sequence'][0]['totalQuantizedSteps'] != '32':
                return

            m21_score = convert_serialized_notes_to_midi(notelist)

            latent_array = [calc_rhtyhmic_entropy_category(m21_score), calc_unique_pitch_number_category(m21_score),
                            calc_biggest_interval_category(m21_score)]

            score_array = self.get_tensor(m21_score)

            score_seq[item_index] = score_array
            latent_seq[item_index] = latent_array

        for i in range(180, 181):
            if i < 10:
                i_expanded = f"00{i}"
            elif i < 100:
                i_expanded = f"0{i}"
            else:
                i_expanded = i
            note_lists = download_file(f"https://storage.googleapis.com/magentadata/datasets/bach-doodle/bach-doodle.jsonl-00{i_expanded}-of-00192.gz")

            print("Erstelle Datenpunkte...")
            for index, note_list in enumerate(note_lists):
                _create_data_point(index, note_list)
            print("Datenpunkte erstellt.")

        score_seq_without_none = []
        latent_seq_without_none = []
        for index, eintrag in enumerate(score_seq):
            if eintrag is not None:
                score_seq_without_none.append(eintrag)
                latent_seq_without_none.append(latent_seq[index])

        self.score_array = np.array(score_seq_without_none)
        self.latent_array = np.array(latent_seq_without_none)
        print('Number of data points: ', self.score_array.shape[0])

        # self.metadata = {
        #     'title': TITLE,
        #     'description': DESCRIPTION,
        #     'version': VERSION_NUM,
        #     'authors': AUTHORS,
        #     'data': date.today().strftime("%B %d, %Y"),
        #     'latents_names': tuple([key for key in self.latent_dicts.keys()]),
        # }
        np.savez(
            self.dataset_path,
            score_array=self.score_array,
            latent_array=self.latent_array,
            note2index_dict=self.note2index_dict,
            index2note_dict=self.index2note_dict,
            latent_dicts=self.latent_dicts,
            metadata={}
            # metadata=self.metadata
        )

    def get_tensor(self, score: music21.stream.Score) -> Union[np.array, None]:
        """
        Returns the score as a torch tensor
        dimension 32!

        Args:
            score: music21.stream.Score object

        Returns:
            torch.Tensor
        """
        notes = get_notes(score)
        if not is_score_on_ticks(score, TICK_VALUES) or has_overlapping_notes(score):
            return None
        list_note_strings_and_pitches = [(n.nameWithOctave, n.pitch.midi)
                                         for n in notes
                                         if n.isNote]
        for note_name, pitch in list_note_strings_and_pitches:

            if note_name not in self.note2index_dict:
                self.update_index_dicts(note_name)

        # construct sequence
        x = 0
        y = 0
        length = int(score.highestTime * self.beat_subdivisions)  # 8 * 4 = 32
        t = np.zeros(length)

        current_tick = 0

        for x in notes:
            note_time = x.quarterLength * self.beat_subdivisions
            t[current_tick] = self.note2index_dict[standard_name(x)]
            note_time = note_time - 1
            current_tick = current_tick + 1
            while note_time > 0:
                t[current_tick] = self.note2index_dict[CONTINUE_SYMBOL]
                note_time = note_time - 1
                current_tick = current_tick + 1

        lead = t
        lead = lead.astype('int32')
        return lead

    def update_index_dicts(self, new_note_name):
        """
        Updates self.note2index_dicts and self.index2note_dicts
        """
        new_index = len(self.note2index_dict)
        self.index2note_dict.update({new_index: new_note_name})
        self.note2index_dict.update({new_note_name: new_index})
        print(
            f'Warning: Entry {str({new_index: new_note_name})} added to dictionaries'
        )


if __name__ == '__main__':
    data = DoodleDataset()
    data.make_or_load_dataset()
