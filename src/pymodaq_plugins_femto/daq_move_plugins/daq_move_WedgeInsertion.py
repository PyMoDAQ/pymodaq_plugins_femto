from pymodaq.daq_move.utility_classes import DAQ_Move_base  # base class
from pymodaq.daq_move.utility_classes import comon_parameters  # common set of parameters for all actuators
from pymodaq.daq_utils.daq_utils import ThreadCommand, getLineInfo  # object used to send info back to the main thread
from easydict import EasyDict as edict  # type of dict
from pymodaq_plugins.daq_move_plugins.daq_move_MockTau import DAQ_Move_MockTau

class DAQ_Move_WedgeInsertion(DAQ_Move_MockTau):
    """
        Wrapper object to access the Mock fonctionnalities, similar wrapper for all controllers.

        =============== ==============
        **Attributes**    **Type**
        *params*          dictionnary
        =============== ==============
    """
    _controller_units = 'mm'
    is_multiaxes = False
    params = [{'title': 'Wedge angle', 'name': 'wedge_angle', 'type': 'float', 'value': 25}] + DAQ_Move_MockTau.params

    def __init__(self, parent=None, params_state=None):
        super().__init__(parent, params_state)
        self.setting.child('tau').setValue(20)


