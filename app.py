# -*- coding: utf-8 -*-
'''PDF pages counter with GUI.'''
# pyinstaller kaz_clf.spec

"""
open folder – Папканы ашу
browse - Карау
select file - Файлды таңдау
Delete symbols - Таңбаларды жою
Process 1 file - 1 файлды өңдеу
process all file - Барлық файлдарды өңдеу
preview 1 file - 1 файлды алдын ала қарау
Classes - Классы
Words - Сөздер
Phrases - Фразалар
Row - Қатар
Name - Аты
Class - Класс
probability - Ықтималдығы
save as - Сақтау
position - Орналасуы
head word - Бас сөз
child word - Бағынышты сөз (подчиненное слово)
"""

import os.path
import pickle
import sys
import traceback
from string import punctuation
from typing import List, Optional, Tuple

import docx2txt
import fitz
import numpy as np
import PySimpleGUI as sg
import spacy_udpipe
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import _VectorizerMixin
from spacy_udpipe.utils import Language

try:
    ROOT_PATH = sys._MEIPASS
except:
    ROOT_PATH = '.'

LATIN = list('abcdefghijklmnopqrstuvwhyz')
DIGITS = [str(_) for _ in '1234567890']
PUNCT = list(punctuation + '№')
EXCL = set(DIGITS + LATIN + PUNCT)

def open_file(file_name: str) -> str:
    """Open file."""

    content = ''
    if file_name.endswith('.pdf'):
        fd = fitz.open(file_name)
        for page in fd:
            t = page.get_text()
            content += '\n' + t

    elif file_name.endswith('docx'):
        content = docx2txt.process(file_name)

    elif file_name.endswith('.txt'):
        with open(file_name, 'r') as f:
            content = f.read()

    content = content.encode('utf-8', 'replace').decode()

    return content


def cleanup_text(text: str, del_symbols: List[str]) -> str:
    """
    Cleanup text. Delete all `del_symbols`.

    Args:
        text (str): raw text
        del_symbols (List[str]): what o delete. e.g. ['\\n']

    Returns:
        str: pretty clean text
    """

    result = text

    for ds in del_symbols:
        result = result.replace(ds, '')

    return result


def tokenizer(
            text: str,
            nlp_ud: Language,
            filter_pos: Optional[List[str]] = ['NOUN'],
            lemmatize: bool = True,
            ) -> List[str]:
    """
    Tokenize text.

    Args:
        text (str): text
        filter_pos (Optional[List[str]], optional): List of POS tags to iclude. If None - all POS. Defaults to 'NOUN'.

    Returns:
        List[str]: List of tokens.
    """

    doc = nlp_ud(text)
    tokens: List[str] = []
    for token in doc:
        text = token.text
        if lemmatize:
            text = token.lemma_
        pos = token.pos_
        apply = False
        if len(set(text.lower()).intersection(EXCL)) == 0:
            if filter_pos is None:
                apply = True
            elif pos in filter_pos:
                apply = True

        if apply:
            tokens.append(text)

    return tokens


class Tokenizer():
    def __init__(self,
                 nlp_ud,
                 filter_pos,
                 lemmatize) -> None:
        self.nlp_ud = nlp_ud
        self.filter_pos = filter_pos
        self.lemmatize = lemmatize

    def __call__(self, text: str) -> List[str]:
        return tokenizer(
            text=text,
            nlp_ud=self.nlp_ud,
            filter_pos=self.filter_pos,
            lemmatize=self.lemmatize
        )


def classify_text(texts: List[str], vectorizer: _VectorizerMixin, clf: BaseEstimator) -> Tuple[List[float], List[str]]:
    """
    Get text class.

    Args:
        text (List[str]): texts
        vectorizer (_VectorizerMixin): vectorizer (TFiDF or ContVectorizer)
        clf (BaseEstimator): classifier

    Returns:
        Tuple[List[float], List[str]]: probas, labels
    """

    xs = vectorizer.transform(texts)
    ys_proba = clf.predict_proba(xs)
    max_idxs = np.argmax(ys_proba, axis=1)
    all_labels = clf.classes_
    labels: List[str] = []
    probas: List[float] = []
    for i, idx in enumerate(max_idxs):
        labels.append(all_labels[idx])
        probas.append(round(ys_proba[i][idx], 2))

    return probas, labels

def analyze(texts: List[str], nlp_ud: Language):
    """Считает токены и словосочетания."""

    ALLOWED_POS = ['NOUN', 'VERB', 'ADJ', 'ADV']

    lemmas: List[str] = []
    words: List[str] = []
    pos: List[str] = []

    head_lemmas: List[str] = []
    head_words: List[str] = []
    head_pos: List[str] = []
    child_lemmas: List[str] = []
    child_words: List[str] = []
    child_pos: List[str] = []

    for text in texts:
        doc = nlp_ud(text)
        for token in doc:
            if (len(set(token.text.lower()).intersection(EXCL)) == 0
                and token.pos_ in ALLOWED_POS):
                lemmas.append(token.lemma_)
                pos.append(token.pos_)
                words.append(token.text)

                for child in token.children:
                    if (len(set(child.text.lower()).intersection(EXCL)) == 0
                        and child.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']
                        and child.text != token.text):
                        head_lemmas.append(token.lemma_)
                        head_words.append(token.text)
                        head_pos.append(token.pos_)
                        child_lemmas.append(child.lemma_)
                        child_words.append(child.text)
                        child_pos.append(child.pos_)

    result_words = [[*d] for d in zip(words, lemmas, pos)]
    result_phrases = [ [*d] for d in zip(head_words, head_lemmas, head_pos, child_words, child_lemmas, child_pos)]

    return result_words, result_phrases


def main():
    """Main function."""

    result_data_clf = []
    result_csv_text_clf = ''
    # name, class, probability
    result_header_clf = ['аты', 'класс', 'ықтималдығы']
    result_col_width_clf = [20, 20, 10]

    result_data_words = []
    result_csv_text_words = ''
    # 'word', 'lemma', 'pos'
    result_header_words = ['сөз', 'бастапқы пішіні', 'сөйлеу бөлігі']
    result_col_width_words = [20, 20, 10]

    result_data_phrases = []
    result_csv_text_phrases = ''
    # 'head_word', 'head_lemma', 'head_pos', 'child_word', 'child_lemma', 'child_pos'
    result_header_phrases = ['бас сөз', 'негізгі сөздің бастапқы түрі', 'негізгі сөздің сөйлем мүшесі', 'бағынышты сөз', 'бағыныңқы сөздің бастауыш түрі', 'бағыныңқы сөздің сөйлем мүшесі']
    result_col_width_phrases = [20, 20, 10, 20, 20, 10]

    previous_folder=sg.user_settings_get_entry('-folder-', None)
    file_names = []

    try:
        sg.theme('Green Tan')

        nlp_ud = spacy_udpipe.load_from_path(lang='tr',
                                        path=f'{ROOT_PATH}/models/kazakh-ud-2.0-170801.udpipe',
                                        meta={"description": "Custom 'kz' model"})
        nlp_ud.max_length = 1500000 # max text length

        FILTER_POS = ['NOUN', 'VERB', 'ADJ']
        LEMMATIZE = True
        vectorizer_fn = f'{ROOT_PATH}/models/CountVectorizer_NOUN_VERB_ADJ_True.pickle'
        vectorizer: _VectorizerMixin = pickle.load(open(vectorizer_fn, 'rb'))
        vectorizer.tokenizer = Tokenizer(nlp_ud=nlp_ud, filter_pos=FILTER_POS, lemmatize=LEMMATIZE)

        clf_fn = f'{ROOT_PATH}/models/CV_NOUN_VERB_ADJ_True_GradientBooster.pickle'
        clf: BaseEstimator = pickle.load(open(clf_fn, 'rb'))

        file_list_column = [
            [
                sg.Text('Папканы ашу:'),  # open folder
                sg.Input(
                    size=(25, 1),
                    default_text=previous_folder,
                    visible=False,
                    # do_not_clear=False,
                    enable_events=True,
                    key='-FOLDER-'
                ),
                sg.FolderBrowse(initial_folder=previous_folder, button_text='Карау'),
            ],
            [sg.Text('Файлды таңдау:')],  # select file
            [
                sg.Listbox(
                    values=file_names, enable_events=True, size=(20, 20), key='-FILE-LIST-'
                )
            ],
            [sg.Text('Таңбаларды жою:')],  # delete symbols
            [
                sg.Multiline(default_text='\\n\n\\r\n\\t\n\\xa0', size=(20, 10), key='-DEL-SYMBOLS-')
            ],
            [sg.Button('1 файлды өңдеу', key='-OPEN-ONE-')],  # process 1 file
            [sg.Button('Барлық файлдарды өңдеу', key='-OPEN-ALL-')],  # process all files
            [sg.Button('1 файлды алдын ала қарау', key='-PREVIEW-ONE-')],  # preview 1 file
        ]

        v_result_clf = [
            [sg.Text('Нәтиже')],  # classes
            [sg.ProgressBar(max_value=0, visible=False, key='-PB-')],
            [sg.Table(
                values=result_data_clf,
                headings=result_header_clf,
                def_col_width=10,
                # max_col_width=30,
                col_widths=result_col_width_clf,
                auto_size_columns=False,
                display_row_numbers=True,
                justification='right',
                num_rows=41,
                alternating_row_color='lightyellow',
                key='-RESULT-TABLE-CLF-',
                selected_row_colors='red on yellow',
                enable_events=True,
                expand_x=True,
                expand_y=True,
                enable_click_events=True,
                tooltip='Нәтиже'  # result
            )],
            [
                sg.InputText(
                    default_text='',
                    visible=False,
                    do_not_clear=False,
                    enable_events=True,
                    key='-OUT-FILE-CLF-'
                ),
                sg.FileSaveAs(
                    button_text='Сақтау...',  # save as
                    key='-SAVE-CLF-',
                    disabled=True,
                    file_types=(('CSV Files', '*.csv'),),
                    default_extension='*.csv'
                )
            ],
        ]

        v_result_clf[2][0].RowHeaderText = 'Қатар'

        v_result_words = [
            [sg.Text('Сөздер')],  # words
            [sg.ProgressBar(max_value=0, visible=False, key='-PB-')],
            [sg.Table(
                values=result_data_words,
                headings=result_header_words,
                def_col_width=10,
                # max_col_width=30,
                col_widths=result_col_width_words,
                auto_size_columns=False,
                display_row_numbers=True,
                justification='right',
                num_rows=41,
                alternating_row_color='lightyellow',
                key='-RESULT-TABLE-WORDS-',
                selected_row_colors='red on yellow',
                enable_events=True,
                expand_x=True,
                expand_y=True,
                enable_click_events=True,
                tooltip='Нәтиже'  # result
            )],
            [
                sg.InputText(
                    default_text='',
                    visible=False,
                    do_not_clear=False,
                    enable_events=True,
                    key='-OUT-FILE-WORDS-'
                ),
                sg.FileSaveAs(
                    button_text='Сақтау...', # save as
                    key='-SAVE-WORDS-',
                    disabled=True,
                    file_types=(('CSV Files', '*.csv'),),
                    default_extension='*.csv'
                )
            ],
        ]

        v_result_words[2][0].RowHeaderText = 'Қатар'

        v_result_phrases = [
            [sg.Text('Фразалар')],  # phrases
            [sg.ProgressBar(max_value=0, visible=False, key='-PB-')],
            [sg.Table(
                values=result_data_phrases,
                headings=result_header_phrases,
                def_col_width=10,
                # max_col_width=30,
                col_widths=result_col_width_phrases,
                auto_size_columns=False,
                display_row_numbers=True,
                justification='right',
                num_rows=41,
                alternating_row_color='lightyellow',
                key='-RESULT-TABLE-PHRASES-',
                selected_row_colors='red on yellow',
                enable_events=True,
                expand_x=True,
                expand_y=True,
                enable_click_events=True,
                tooltip='Нәтиже'  # result
            )],
            [
                sg.InputText(
                    default_text='',
                    visible=False,
                    do_not_clear=False,
                    enable_events=True,
                    key='-OUT-FILE-PHRASES-'
                ),
                sg.FileSaveAs(
                    button_text='Сақтау...',  # save as
                    key='-SAVE-PHRASES-',
                    disabled=True,
                    file_types=(('CSV Files', '*.csv'),),
                    default_extension='*.csv'
                )
            ],
        ]

        v_result_phrases[2][0].RowHeaderText = 'Қатар'



        res_tab = [[sg.TabGroup([
                                [  sg.Tab('Классы', v_result_clf),  # classes
                                   sg.Tab('Сөздер', v_result_words),  # words
                                   sg.Tab('Фразалар', v_result_phrases),  # phrases
                                ]])
               ]]

        layout = [
            [
                sg.Column(file_list_column),
                sg.VSeperator(),
                sg.Column(res_tab),
            ],
        ]

        window = sg.Window(
            title='Kazakh classifier',
            icon=f'{ROOT_PATH}/icon.ico',
            layout=layout,
            enable_close_attempted_event=True,
            location=sg.user_settings_get_entry('-location-', (None, None))
        )


        while True:
            event, values = window.read()

            if event in ('Exit', sg.WIN_CLOSED, sg.WINDOW_CLOSE_ATTEMPTED_EVENT):
                # save settings
                sg.user_settings_set_entry('-location-', window.current_location())
                sg.user_settings_set_entry('-folder-', values['-FOLDER-'])
                break

            if event == '-FOLDER-':
                folder = values['-FOLDER-']
                try:
                    file_list = os.listdir(folder)
                except:
                    file_list = []

                file_names = [
                    f
                    for f in file_list
                    if os.path.isfile(os.path.join(folder, f))
                    and f.lower().endswith(('.pdf', '.docx', '.txt'))
                ]
                window['-FILE-LIST-'].update(file_names)

            if event == '-PREVIEW-ONE-':
                del_symbols = values['-DEL-SYMBOLS-']
                del_symbols = del_symbols.split('\n')

                if len(values["-FILE-LIST-"]):
                    f_name = values["-FILE-LIST-"][0]
                    f_path = os.path.join(
                        values["-FOLDER-"], f_name
                    )
                    text = open_file(f_path)
                    text = cleanup_text(text, del_symbols)

                    sg.popup_scrolled(text, size=(100, 50), title='Preview')

            if event == '-OPEN-ONE-':
                if len(values['-FILE-LIST-']):
                    del_symbols = values['-DEL-SYMBOLS-']
                    del_symbols = del_symbols.split('\n')
                    f_name = values['-FILE-LIST-'][0]
                    f_path = os.path.join(values['-FOLDER-'], f_name)

                    text = open_file(f_path)
                    text = cleanup_text(text, del_symbols)

                    if text != '':
                        probas, labels = classify_text([text], vectorizer=vectorizer, clf=clf)
                        result_data_clf = [[f_name, labels[0], probas[0]]]

                        res_words, res_phrases = analyze([text], nlp_ud=nlp_ud)
                        result_data_words = res_words
                        result_data_phrases = res_phrases
                        window['-RESULT-TABLE-CLF-'].update(values=result_data_clf)
                        window['-RESULT-TABLE-WORDS-'].update(values=result_data_words)
                        window['-RESULT-TABLE-PHRASES-'].update(values=result_data_phrases)
                        window['-SAVE-CLF-'].update(disabled=False)
                        window['-SAVE-WORDS-'].update(disabled=False)
                        window['-SAVE-PHRASES-'].update(disabled=False)

            if event == '-OPEN-ALL-':
                if len(file_names) > 0:
                    del_symbols = values['-DEL-SYMBOLS-']
                    del_symbols = del_symbols.split('\n')
                    result_data_clf = []
                    p_bar = window['-PB-']
                    pb_max = len(file_names)
                    pb_max = pb_max + 6
                    pb_step = 3
                    f_names = []
                    texts = []

                    for i, f_name in enumerate(file_names):
                        if pb_step >= pb_max and i % pb_step == 0:
                            p_bar.update(current_count=i, max=pb_max, visible=True)

                        f_path = os.path.join(values['-FOLDER-'], f_name)

                        text = open_file(f_path)
                        text = cleanup_text(text, del_symbols)

                        if text != '':
                            f_names.append(f_name)
                            texts.append(text)

                    if len(f_names) > 0:
                        probas, labels = classify_text(texts, vectorizer=vectorizer, clf=clf)
                        result_data_clf  = [[f_names[i], l, probas[i]] for i, l in enumerate(labels)]

                        res_words, res_phrases = analyze(texts, nlp_ud=nlp_ud)
                        result_data_words = res_words
                        result_data_phrases = res_phrases

                        window['-RESULT-TABLE-CLF-'].update(values=result_data_clf)
                        window['-RESULT-TABLE-WORDS-'].update(values=result_data_words)
                        window['-RESULT-TABLE-PHRASES-'].update(values=result_data_phrases)
                        window['-SAVE-CLF-'].update(disabled=False)
                        window['-SAVE-WORDS-'].update(disabled=False)
                        window['-SAVE-PHRASES-'].update(disabled=False)

                        p_bar.update(current_count=pb_max, max=pb_max, visible=True)

                    p_bar.update(visible=False)

            # click on table
            if event == '-RESULT-TABLE-CLF-':
                rows = values[event]
                if len(rows) > 0 and len(result_data_clf) > 0:
                    row_i = rows[0]
                    if len(result_data_clf) > 0:
                        row = result_data_clf[row_i]
                        text = '\n'.join([f'{col}:\t{row[i]}' for i, col in enumerate(result_header_clf)])
                        sg.popup_scrolled(text, size=(100, 10), title='Қатар')

            if event == '-OUT-FILE-CLF-':
                out_file = values['-OUT-FILE-CLF-']

                # construct csv
                result_csv = []
                result_csv.append(';'.join(result_header_clf))
                for r in result_data_clf:
                    result_csv.append(';'.join([f'"{t}"' for t in r]))

                result_csv_text_clf = '\n'.join(result_csv)

                if result_csv_text_clf != '' and out_file != '':
                    with open(out_file, 'wt', encoding='UTF-8') as f:
                        f.write(result_csv_text_clf)
                        sg.popup_ok(f'File was saved: "{out_file}"')

            if event == '-OUT-FILE-WORDS-':
                out_file = values['-OUT-FILE-WORDS-']

                # construct csv
                result_csv = []
                result_csv.append(';'.join(result_header_words))
                for r in result_data_words:
                    result_csv.append(';'.join([f'"{t}"' for t in r]))

                result_csv_text_words = '\n'.join(result_csv)

                if result_csv_text_words != '' and out_file != '':
                    with open(out_file, 'wt', encoding='UTF-8') as f:
                        f.write(result_csv_text_words)
                        sg.popup_ok(f'File was saved: "{out_file}"')

            if event == '-OUT-FILE-PHRASES-':
                out_file = values['-OUT-FILE-PHRASES-']

                # construct csv
                result_csv = []
                result_csv.append(';'.join(result_header_phrases))
                for r in result_data_phrases:
                    result_csv.append(';'.join([f'"{t}"' for t in r]))

                result_csv_text_phrases = '\n'.join(result_csv)

                if result_csv_text_phrases != '' and out_file != '':
                    with open(out_file, 'wt', encoding='UTF-8') as f:
                        f.write(result_csv_text_phrases)
                        sg.popup_ok(f'File was saved: "{out_file}"')

        window.close()

    except Exception as e:
        tb = traceback.format_exc()
        sg.Print(f'An error happened.  Here is the info:', e, tb)
        sg.popup_error(f'AN EXCEPTION OCCURRED!', e, tb)



if __name__ == '__main__':
    main()
