from torch import load as load_torch
from torch import save as save_torch
import os
from torch import is_tensor
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
import PySimpleGUI as sg
from DTGUILib import utils as dtu
from DTGUILib import elements as dte
from DTGUILib import const as dtc
import gc

from torch.cuda import empty_cache

sg.theme('DarkGrey7')

def remove_key_from_dict(d, key):
    enumerated_key = None
    
    # Check if the key is an enumerated key and store the base key
    if "_" in key:
        base_key = key.rsplit("_", 1)[0]
        if base_key in d:
            enumerated_key = base_key
            
    if key in d:
        # Handle regular keys
        del d[key]
        return True
    elif enumerated_key:
        # Handle enumerated keys
        keys_to_remove = [k for k in d if k.startswith(enumerated_key)]
        for k in keys_to_remove:
            del d[k]
        return True
    return False

def remove_nested_key(obj, keys_to_remove):
    if not keys_to_remove:
        return

    key = keys_to_remove[0]
    if isinstance(obj, dict):
        if key in obj:
            if len(keys_to_remove) == 1:
                del obj[key]
            else:
                remove_nested_key(obj[key], keys_to_remove[1:])
    elif isinstance(obj, list):
        index = int(key)
        if index < len(obj):
            if len(keys_to_remove) == 1:
                obj.pop(index)
            else:
                remove_nested_key(obj[index], keys_to_remove[1:])

def ratio_merge(model_paths, alphas=None, matchwords=None, device='cpu', roots=['state_dict']):
    gc.collect()
    empty_cache()

    if alphas is None:
        alphas = [1 / len(model_paths)] * len(model_paths)
    if matchwords is None:
        matchwords = []

    # Initialize merged model
    merged_model = {root: {} for root in roots}
    merged_initialized = False

    # Load and merge models
    for model_path, alpha in zip(model_paths, alphas):
        model = load_torch(model_path, map_location=device)
        roots = ['ALL'] if not roots else roots
        for root in roots:
            theta = model[root] if root != 'ALL' else model
            insert = merged_model[root] if root != 'ALL' else merged_model       
            for key in theta.keys():
                # Check if key matches any of the matchwords or if matchwords is empty
                if any(matchword in key for matchword in matchwords) or not matchwords:
                    if key in insert:
                        insert[key] += alpha * theta[key]
                    else:
                        if not merged_initialized:
                            dtu.log(f"Key {key} not found in merged model. Adding it now.", lvl='warning')
                        insert[key] = alpha * theta[key]
        if not merged_initialized:
            merged_initialized = True

        del model

    return merged_model




def create_expandable_element(key, input_pth=''):
    return [
        sg.Input(default_text=(input_pth if key==0 else ''), key=f'ckpt_path_{key}', size=(40, 1), visible=(key == 0)),
        sg.FileBrowse('Browse', key=f'ckpt_path_{key}_btn', target=f'ckpt_path_{key}', visible=(key == 0)),
        sg.Text('Ratio:', key=f'ratio_text_{key}', visible=(key == 0)),
        sg.Input('0.5', key=f'ratio_{key}', size=(5, 1), visible=(key == 0)),
        sg.Button('+', key=f'add_{key}', visible=(key == 0))
    ]

def expand(i, window):
    window[f'add_{i}'].update(visible=False)
    window[f'ckpt_path_{i+1}'].update(visible=True)
    window[f'ckpt_path_{i+1}_btn'].update(visible=True)
    window[f'ratio_text_{i+1}'].update(visible=True)
    window[f'ratio_{i+1}'].update(visible=True)
    window[f'add_{i+1}'].update(visible=True) 

def merger_popup(input_pth, main_window, loaded_dict):
    num_elements = 25
    layout = [
        create_expandable_element(i, input_pth) for i in range(num_elements)
    ] + [
        dte.String_Setting(text='Matchwords (comma sep):', key='matchwords', default='main'),
        dte.String_Setting(text='Roots to process (comma sep):', key='roots', default='state_dict'),
        dte.List_Setting(text='Device:', list=['cpu', 'cuda', 'mps'], key='device', default='cpu'),
        [sg.Button('Submit'), sg.Button('Cancel')],
    ]
    
    merge_window = sg.Window('Merger', layout, finalize=True)
    expand(0, merge_window)
    
    while True:
        event, values = merge_window.read()
    
        if event == sg.WIN_CLOSED or event == 'Cancel':
            merge_window.close()
            return None
    
        for i in range(num_elements):
            if event == f'add_{i}':
                expand(i, merge_window)
                break
        else:
            if event == 'Submit':
                values = {k: v for k, v in values.items() if v != ''}
                checkpoints = []
                ratios = []
                for v in values.keys():
                    if v.startswith('ckpt_path_'):
                        checkpoints.append(values[v])
                    elif v.startswith('ratio_'):
                        ratios.append(float(values[v]))
                ratios = ratios[:len(checkpoints)]
                matchwords = '' if 'matchwords' not in values else values['matchwords']
                roots = '' if 'roots' not in values else values['roots']
                merged = ratio_merge(checkpoints, ratios, matchwords.split(', '), values['device'], roots.split(', '))
                fill_tree(merged)
                merge_window.close()

loaded_dict = {}
def get_supported_files():
    return (("Torch, Safetensors Files", "*.ckpt *.pt *.pth *.safetensors"),)

layout = [
    [sg.Menubar([['File', ['Open::-OPEN-', 'Save::-SAVE-']]], key='-MENUBAR-')],
    [sg.Tree(data=dte.TreeData(), headings=['Value'], auto_size_columns=True, num_rows=20, col0_width=40, key='tree', show_expanded=False, enable_events=True, expand_x=True, expand_y=True)],    
    [sg.Button('Merger')],
    [dte.ConsoleClass(size=(5,10), expand_x=True, route_err=False)]
]


window = sg.Window("TorchSpector", layout, finalize=True, enable_close_attempted_event=True, resizable=True, size=(800, 600), icon=dtc.DT_ICON)


def load_file(file, fill=True):
    global loaded_dict
    global loaded_path
    if file.endswith('.safetensors'):
        data_dict = load_safetensors(file)
    else:
        data_dict = load_torch(file, map_location='cpu')
    if fill:
        dtu.log(f"Loaded file, now filling values recursively..", lvl='loading')
        fill_tree(data_dict)
    loaded_dict = data_dict
    loaded_path = file
    return data_dict

def fill_tree(data_dict):
    # clear tree
    treedata = dte.TreeData()
    keys_seen = {}
    for rootkey, value in data_dict.items():
        process_key_values('', rootkey, value, treedata, keys_seen)
    window['tree'].update(treedata)    

def process_key_values(rootkey, key, value, treedata, keys_seen):
    if isinstance(value, dict):
        # go down if its dict
        if key in keys_seen:
            new_key = f"{key}_{keys_seen[key]}"
            keys_seen[key] += 1
        else:
            new_key = key
            keys_seen[key] = 1
        treedata.Insert(parent=rootkey, key=new_key, text=key, values=[])
        for k, v in value.items():
            process_key_values(new_key, k, v, treedata, keys_seen)

    elif isinstance(value, list):
        # go down if its list
        if key in keys_seen:
            new_key = f"{key}_{keys_seen[key]}"
            keys_seen[key] += 1
        else:
            new_key = key
            keys_seen[key] = 1
        treedata.Insert(parent=rootkey, key=new_key, text=key, values=[])
        for i, v in enumerate(value):
            process_key_values(new_key, f"{key}_{i}", v, treedata, keys_seen)
    elif is_tensor(value):
        # add tensor shape if it is a tensor
        if key in keys_seen:
            new_key = f"{key}_{keys_seen[key]}"
            keys_seen[key] += 1
        else:
            new_key = key
            keys_seen[key] = 1
        treedata.Insert(parent=rootkey, key=new_key, text=key, values=[str(tuple(value.shape))])
    else:
        if key in keys_seen:
            new_key = f"{key}_{keys_seen[key]}"
            keys_seen[key] += 1
        else:
            new_key = key
            keys_seen[key] = 1
        treedata.Insert(parent=rootkey, key=new_key, text=key, values=[value])


dtu.log('Checkpoint files may contain malicious code. Only load checkpoints from trusted sources.', lvl='nonurgent')

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSE_ATTEMPTED_EVENT, 'Exit'):
        break

    if event == 'Open::-OPEN-':
        open_file_name = ''
        if sg.running_mac():
            open_file_name = sg.tk.filedialog.askopenfilename(defaultextension='.ckpt')  # show the 'get files' dialog box
        else:
            open_file_name = sg.tk.filedialog.askopenfilename(filetypes=get_supported_files(), defaultextension='.ckpt')  # show the 'get files' dialog box

        if open_file_name:
            dtu.log(f"Loading {open_file_name}", lvl='loading')
            window.start_thread(lambda: load_file(open_file_name), 'DONELOAD')   
    if event == 'DONELOAD':
        dtu.log(f"Checkpoint loaded successfully!", lvl='success')

    if event == 'Merger':
        if loaded_dict:
            merged = merger_popup(loaded_path, window, loaded_dict)
        else:
            merged = merger_popup('', window, loaded_dict)



    # if loaded
    if loaded_dict:
        if event == 'Save::-SAVE-':
            input_filename = os.path.basename(loaded_path)
            input_dirname = os.path.dirname(loaded_path)
            if sg.running_mac():
                save_file_name = sg.tk.filedialog.asksaveasfilename(defaultextension='.ckpt', initialdir=input_dirname, initialfile=input_filename)
            else:
                save_file_name = sg.tk.filedialog.asksaveasfilename(filetypes=get_supported_files(), defaultextension='.ckpt', initialdir=input_dirname, initialfile=input_filename)
            if save_file_name:
                if save_file_name.endswith('.safetensors'):
                    save_safetensors(loaded_dict, save_file_name)
                else:
                    save_torch(loaded_dict, save_file_name)
                dtu.log(f"Saved checkpoint to {save_file_name}", lvl='success')

window.close()