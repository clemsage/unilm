"""
    This file gathers methods specific to the the Scanned Receipts OCR and Information Extraction (SROIE) 2019 dataset
    (see https://rrc.cvc.uab.es/?ch=13 for more information).
"""
import argparse
import os
import json
import numpy as np
from PIL import Image, ImageDraw
from collections import OrderedDict, Counter
import editdistance


def prepare_dataset(args):
    """
    Prepare the training, development and test subsets from the SROIE 2019 original files, including pre-processing
    for the input text and for the output labels.

    Args:
        args: parsed arguments
    Returns:
        None
    """
    np.random.seed(args.seed)

    folders = {'text': {'training': '0325updated.task1train(626p)', 'test': 'text.task1_2-test（361p)'},
               'image': {'training': '0325updated.task1train(626p)', 'test':  os.path.join('task3-test 347p) -',
                                                                                           'task3-test（347p)')},
               'ground_truth': {'training': '0325updated.task2train(626p)', 'test': None}
               }  # Ground truth is not provided for the test set

    document_ids = {  # filenames are X[0-9]{11}\([0-9]+\).(jpg|txt)
        set_type: list(OrderedDict.fromkeys(
            file_path.split('.')[0] for file_path in sorted(os.listdir(os.path.join(args.raw_dataset_path, folder)))
            if '(' not in file_path))  # to discard duplicate documents
        for set_type, folder in folders['image'].items()}

    # verify that we have retrieved all elements of the training and test sets
    assert len(document_ids['training']) == 626
    assert len(document_ids['test']) == 347
    assert len(set(document_ids['training']).intersection(set(document_ids['test']))) == 0

    assert all([os.path.exists(os.path.join(
        args.raw_dataset_path, folders['ground_truth']['training'], ''.join((doc_id, '.txt'))))
        for doc_id in document_ids['training']])  # check we have the ground truth for every training document

    for set_type in ['training', 'test']:  # check we have the text for every document
        assert all([os.path.exists(os.path.join(
            args.raw_dataset_path, folders['text'][set_type], ''.join((doc_id, '.txt'))))
            for doc_id in document_ids[set_type]])

    # we set aside a small number of training docs to constitute a development set
    nb_devl_docs = 26
    np.random.shuffle(document_ids['training'])
    document_ids.update({'training': document_ids['training'][nb_devl_docs:],
                         'development': document_ids['training'][:nb_devl_docs]})
    for elt in folders.values():
        elt['development'] = elt['training']  # devl shares the same folders as train

    print('Preparing the SROIE dataset for sequence labeling...')
    index_dict = {}  # create a index file for rapidly retrieving documents of each set
    for set_type in ['training', 'development', 'test']:
        index_dict[set_type] = []
        print('\n======== %s =========' % set_type)
        prepared_set_path = os.path.join(args.dataset_path, set_type)
        os.makedirs(os.path.join(prepared_set_path, 'annotations'), exist_ok=True)
        os.makedirs(os.path.join(prepared_set_path, 'images'), exist_ok=True)
        nb_processing_errors = Counter()
        for doc_id in document_ids[set_type]:
            index_dict[set_type].append(os.path.join(set_type, doc_id))
            # text
            filename = os.path.join(args.raw_dataset_path, folders['text'][set_type], ''.join((doc_id, '.txt')))
            try:
                text = [line.decode('utf-8').rstrip() for line in open(filename, 'rb')]
            except UnicodeDecodeError:
                text = [line.decode('iso-8859-1').rstrip() for line in open(filename, 'rb')]

            # image
            filename = os.path.join(args.raw_dataset_path, folders['image'][set_type], ''.join((doc_id, '.jpg')))
            image = Image.open(filename)

            receipt_json = pre_process_input(text, image)

            if folders['ground_truth'][set_type] is not None:
                filename = os.path.join(args.raw_dataset_path, folders['ground_truth'][set_type],
                                        ''.join((doc_id, '.txt')))
                ground_truth = json.load(open(filename, 'rb'))
                nb_processing_errors_in_doc = pre_process_output(ground_truth, receipt_json, doc_id, image,
                                                                 args.with_heuristics_for_word_labels)
                nb_processing_errors.update(nb_processing_errors_in_doc)

            # save the pre-processed input and output in a JSON file
            with open(os.path.join(prepared_set_path, 'annotations', '%s.json' % doc_id), 'w') as document_f:
                json.dump(receipt_json, document_f)

            # save the image
            image.save(os.path.join(prepared_set_path, 'images', '%s.png' % doc_id))

        print('Number of incoherent ground truth for the %s set: %s' % (set_type, nb_processing_errors))

    index_path = os.path.join(args.dataset_path, 'datasets_seed_%d.txt' % args.seed)
    with open(index_path, 'w') as f:
        json.dump(index_dict, f)
    print('Document index for each subset of the dataset has been saved in the file %s' % index_path)


def pre_process_input(text, image, draw_word_bb=False, min_overlapping_line=0.5):
    """
    Pre-process the provided text by reordering the text segments from top-to-bottom and left-to-right and dividing the
    segments into words (the separators between words are spaces and ':')

    Args:
        text: list of strings corresponding to the text segments of a receipt
        image: PIL.Image.Image corresponding to the receipt image
        draw_word_bb: draw and show the estimated bounding boxes of words for verification purposes
        min_overlapping_line: minimum overlap of a text segment with a given line for considering that the segment belongs
                              to the line
    Returns:
        receipt_json: nested dict filled with pre-processed input of a receipt
    """
    if draw_word_bb:
        if image.mode == 'L':  # grayscale colormap
            image = image.convert('RGB')
        draw = ImageDraw.Draw(image, 'RGBA')
    else:
        draw = None

    receipt_json = {'form': []}

    # reorder text segments from top to bottom as provided order is sometimes wrong
    lines = []
    insertion_idx = 0  # to create the first line with the first segment
    for segment in text:
        if segment == '':  # discard empty segment
            continue
        elements = segment.split(',')
        segment = {'left': int(elements[0]), 'top': int(elements[1]),
                   'right': int(elements[4]), 'bottom': int(elements[5]),
                   'text': ','.join(elements[8:]).strip()}

        # aggregate segments in lines, sorted by vertical positions
        for current_line_idx, line in enumerate(lines):
            if segment['bottom'] < line['top'] + min_overlapping_line * (line['bottom'] - line['top']):
                insertion_idx = 0
                break
            else:
                # check if the segment belongs to the current line, i.e. occupies at least X % of the line height
                # and there is no overlap over the x-axis between the line and the segment
                if segment['top'] <= line['bottom'] - min_overlapping_line * (line['bottom'] - line['top'])\
                        and (segment['right'] <= line['left'] or segment['left'] >= line['right']):
                    insertion_idx = None
                    line['top'] = min(line['top'], segment['top'])  # update line coordinates
                    line['bottom'] = max(line['bottom'], segment['bottom'])
                    line['left'] = min(line['left'], segment['left'])
                    line['right'] = max(line['right'], segment['right'])
                    line['segments'].append(segment)
                    break
                else:
                    if current_line_idx + 1 < len(lines):  # check if there is a line below the current one
                        if segment['bottom'] < lines[current_line_idx + 1]['top'] + min_overlapping_line * \
                                (lines[current_line_idx + 1]['bottom'] - lines[current_line_idx + 1]['top']):
                            insertion_idx = current_line_idx + 1
                            break
                    else:
                        insertion_idx = current_line_idx + 1
                        break

        if insertion_idx is not None:
            line = {'top': segment['top'], 'bottom': segment['bottom'], 'left': segment['left'],
                    'right': segment['right'], 'segments': [segment]}
            lines.insert(insertion_idx, line)

    # reorder segments among each line following left-to-right order
    segments = []
    for idx_line, line in enumerate(lines):
        while len(line['segments']):
            i = min(list(range(len(line['segments']))), key=lambda x: line['segments'][x]['left'])
            segment = line['segments'][i]
            segment['idx_line'] = idx_line
            segments.append(segment)
            del line['segments'][i]

    for segment in segments:
        # convert the text segment to words
        space_by_char = (segment['right'] - segment['left']) / len(segment['text'])
        words = segment["text"].split()
        words = [w for w in words if w != '']
        x_offset = 0
        for i, word in enumerate(words):
            word_width = len(word) * space_by_char
            word = {'left': int(x_offset + segment['left']), 'right': int(x_offset + segment['left'] + word_width),
                    'top': segment['top'], 'bottom': segment['bottom'], 'token': word, 'idx_line': segment['idx_line']}
            x_offset += word_width

            if word['token'] != ' ':  # remove spaces from list of words
                receipt_json['form'].append({
                    'label': "other",  # by default, the item does not carry information
                    'words': [
                        {
                            'box': [word['left'], word['top'], word['right'], word['bottom']],
                            'text': word['token']
                        }
                    ],
                    'idx_line': word['idx_line']
                })
                if draw_word_bb:
                    draw.rectangle([word['left'], word['top'], word['right'], word['bottom']], outline='cyan')

    if draw_word_bb:
        image.show()

    return receipt_json


def pre_process_output(ground_truth, receipt_json, doc_id, image, with_heuristics_for_word_labels,
                       max_edit_distance=5, max_nb_words_diff=3):
    """
    Pre-process the output by matching the field values provided in the ground truth with the words of the receipt.
    The matching is permissive to take into account the small character differences between the text and the annotated
    field values. Some errors are also manually corrected.

    Args:
        ground_truth: dict containing the fields to extract and their expected values for the receipt
        receipt_json: nested dict filled with pre-processed input of a receipt and about to be filled with the
                      pre-processed output
        doc_id: string following the pattern X[0-9]{11} identifying the receipt
        image: PIL.Image.Image corresponding to the receipt image
        with_heuristics_for_word_labels: boolean indicating if heuristics are employed for deriving better word level
                                         supervision from provided ground truth.
        max_edit_distance: integer indicating the maximum number of edit operations allowed for matching a field value
                           with the text of the receipt
        max_nb_words_diff: integer indicating the maximum absolute difference of number of words between a field value
                           and candidate words from the receipt
    Returns:
        fields_in_error: list of fields for which we have not found at least one occurrence of the field value
                         among the document words
    """
    fields_in_error = []

    # manually correct annotation errors that are not automatically handled later
    if doc_id == 'X51005447850':  # '20180304' in ground truth file
        ground_truth['date'] = '04/03/2018'
    if doc_id == 'X51008114284':  # 'KAWASAN PERINDUSTRIAN BALAKONG' duplicated in ground truth
        ground_truth['address'] = 'LOT 1851-A & 1851-B, KAWASAN PERINDUSTRIAN BALAKONG, 43300 SERI KEMBANGAN, SELANGOR'
    if doc_id == 'X51005719874':  # Too many misspelling errors in ground truth (8)
        ground_truth['address'] = 'LOT NO . : G1 . 116A , GROUND FLOOR , SUNWAY PYRAMID , NO . 3 , JALAN PJS 11/15 , ' \
                                  'BANDAR SUNWAY , 47500 PETALING JAYA . '
    if doc_id == 'X51008114284':  # Missing 'JALAN KPB 6' in ground truth
        ground_truth['address'] = 'LOT 1851-A & 1851-B, JALAN KPB 6, KAWASAN PERINDUSTRIAN BALAKONG, 43300 SERI ' \
                                  'KEMBANGAN, SELANGOR '
    if doc_id == 'X51006328920':  # Faded receipt
        ground_truth['address'] = 'NO. AN SS4C/5,PETALING JAYA ANGOR DARUL EHSAN'

    # complete ground truth if missing
    for field, field_value in ground_truth.items():
        if field_value == '':
            if doc_id == 'X51005433522' and field == 'total':
                ground_truth[field] = '$8.20'
            else:
                raise Exception('Field %s is empty for document ID: %s' % (field, doc_id))

    # find all field occurrences in the receipt : fuzzy matching with a maximum number of edit operations allowed
    field_occurrences = {field: [] for field in ground_truth.keys()}
    edit_distances = {field: max_edit_distance for field in ground_truth.keys()}
    nb_words_in_field_value = {field: len(field_value.split(' ')) for field, field_value in ground_truth.items()}

    for i in range(len(receipt_json['form'])):
        for field, field_value in ground_truth.items():
            for diff in range(-max_nb_words_diff, max_nb_words_diff + 1):
                nb_words = nb_words_in_field_value[field] + diff
                matching_words = ' '.join([item['words'][0]['text'] for item in receipt_json['form'][i:i + nb_words]])
                dist = editdistance.eval(matching_words, field_value)
                if dist <= edit_distances[field]:
                    if dist < edit_distances[field]:
                        field_occurrences[field] = []
                    if i not in sum(field_occurrences[field], []):  # word already belongs to a occurrence ?
                        field_occurrences[field].append(list(range(i, i + nb_words)))
                        edit_distances[field] = dist

    # check that we have retrieved at least one occurrence of each field instance among document words
    for field in ground_truth.keys():
        try:
            assert len(field_occurrences[field]) > 0
        except AssertionError:
            fields_in_error.append(field)
            image.show()
            raise Exception('For the document %s, we have not found any occurrence of the instance %s of the field %s'
                            % (doc_id, ground_truth[field], field))

    chosen_occurrence = {}
    # choose the most frequent textual value among candidates
    for field, occurrences in field_occurrences.items():
        textual_values, chosen_textual_value = [], []
        for occurrence in occurrences:
            textual_values.append(' '.join([receipt_json['form'][i]['words'][0]['text'] for i in occurrence]))
        chosen_textual_value = Counter(textual_values).most_common(1)[0][0]
        occurrences = [occurrence for i, occurrence in enumerate(occurrences)
                       if textual_values[i] == chosen_textual_value]

        # choose one occurrence of the chosen textual value for deriving word labels
        chosen_occurrence[field] = None
        if with_heuristics_for_word_labels:

            if field == 'total':  # find the bottom most occurrence with the keyword total in its line
                relevant_occurrences = []
                for occurrence in occurrences:
                    idx_line = receipt_json['form'][occurrence[0]]['idx_line']
                    for item in receipt_json['form']:
                        if item['idx_line'] == idx_line:
                            if 'total' in item['words'][0]['text'].lower():
                                relevant_occurrences.append(occurrence)
                if len(relevant_occurrences) == 0:
                    relevant_occurrences = occurrences
                # sort by vertical position within the receipt, the top occurrences appear in the first positions
                relevant_occurrences.sort(key=lambda occ: receipt_json['form'][occ[0]]['words'][0]['box'][-1])
                chosen_occurrence[field] = relevant_occurrences[-1]

        if chosen_occurrence[field] is None:  # randomly
            chosen_occurrence[field] = occurrences[np.random.choice(len(occurrences))]

    # gather the words of each chosen field occurrence
    words_to_remove_from_items = []
    for field, occurrence in chosen_occurrence.items():
        receipt_json['form'][occurrence[0]]['label'] = field
        for idx_word in occurrence[1:]:
            receipt_json['form'][occurrence[0]]['words'].append(receipt_json['form'][idx_word]['words'][0])
            words_to_remove_from_items.append(idx_word)

    receipt_json['form'] = [i for j, i in enumerate(receipt_json['form']) if j not in words_to_remove_from_items]

    return fields_in_error


def populate_candidates_from_predictions(candidates, words, prediction):
    word = words[-1]
    if prediction == 'O':
        return

    prefix, field = prediction.split('-')
    if prefix == 'S':
        candidates[field].append(word)
    elif prefix == 'B':
        candidates[field].append([word])
    elif prefix == 'I':
        if len(candidates[field]) and isinstance(candidates[field][-1], list):
            candidates[field][-1].append(word)
    elif prefix == 'E':
        if len(candidates[field]) and isinstance(candidates[field][-1], list):
            candidates[field][-1] = ' '.join(candidates[field][-1] + [word])
    else:
        raise Exception('Unknown prefix for the BIOES labelling scheme: %s' % prefix)


def predictions_post_processing(args):
    if not args.predictions:
        raise Exception('You must provide the path to the file containing predictions')

    output_folder = os.path.join(os.path.dirname(args.predictions), 'post_processed_predictions')
    os.makedirs(output_folder, exist_ok=True)

    with open(args.predictions, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # get all the fields contained in from predictions
    labels = []
    for line in lines:
        if line.startswith("-DOCSTART-") or line == "" or line == "\n":
            continue
        labels.append(line.split(' ')[1])
    labels = set(labels)
    fields = set(label.split('-')[1] for label in labels if label != 'O')

    print("Outputted fields: %s" % fields)

    def process_current_document():
        final_results = {}
        for field in fields:
            candidates[field] = [' '.join(cand) if isinstance(cand, list) else cand for cand in candidates[field]]
            # choose the most frequent candidate
            cnt = Counter(candidates[field]).most_common(1)
            if len(cnt):
                final_results[field] = cnt[0][0]

        # output post processed predictions in a text file
        final_results = {key.lower(): val for key, val in final_results.items() if key in fields}
        with open(os.path.join(output_folder, current_doc_id + '.txt'), 'w') as f:
            json.dump(final_results, f)

    candidates = {field: [] for field in fields}
    current_doc_id = None
    words = []
    for line in lines:
        if line.startswith("-DOCSTART-") or line == "" or line == "\n":  # introduces a new example
            continue

        word, prediction, doc_id = line.split(' ')

        if current_doc_id is None:
            current_doc_id = doc_id

        if doc_id != current_doc_id:
            process_current_document()
            candidates = {field: [] for field in fields}
            current_doc_id = doc_id
            words = []

        words.append(word)
        populate_candidates_from_predictions(candidates, words, prediction)

    process_current_document()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_pre_processing", default=False, action="store_true")
    parser.add_argument("--do_post_processing", default=False, action="store_true")

    parser.add_argument("--seed", type=int, default=42,
                        help="integer used for randomly creating a development set from the training set")
    parser.add_argument("--with_heuristics_for_word_labels", type=bool, default=True,
                        help="whether to employ heuristics for deriving better word level supervision from provided "
                             "ground truth.")
    parser.add_argument("--raw_dataset_path", type=str, default="../../datasets/SROIE2019/raw",
                        help="path to the folders containing the original dataset")
    parser.add_argument("--dataset_path", type=str, help="path to the resulting pre-processed dataset")

    parser.add_argument("--predictions", type=str, default="", help="path to the predictions")

    args = parser.parse_args()

    if args.do_pre_processing:
        prepare_dataset(args)

    if args.do_post_processing:
        predictions_post_processing(args)
