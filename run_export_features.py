from utils.export import export_features

# for imports within export_features:
import sys
sys.path.append('utils')

export_features(
    data_path = 'my_stimuli/midi/',
    output_path = 'my_stimuli/midi/output',
    #model_name = 'PolyRNN', # can run on cpu
    model_name = 'PolyTNN', # likely needs gpu even to run pre-trained model
)
