import numpy as np
from easydict import EasyDict as edict
from pymodaq.daq_utils.daq_utils import ThreadCommand, getLineInfo, DataFromPlugins, Axis, gauss1D, normalize
from pymodaq.daq_viewer.utility_classes import DAQ_Viewer_base, comon_parameters
from pymodaq.daq_utils.parameter.utils import iter_children, get_param_path
from pathlib import Path
from pypret.frequencies import om2wl, wl2om, convert
from pypret import FourierTransform, Pulse, PNPS, PulsePlot, lib
from pymodaq_femto.simulation import Simulator
from scipy.interpolate import interp2d

class DAQ_1DViewer_Femto(DAQ_Viewer_base):
    """
    """
    params = comon_parameters + \
        [{'title': 'Simulation settings:', 'name': 'simul_settings', 'type': 'group',
            'children': [
                {'title': '', 'name': 'show_pulse_bool', 'type': 'bool_push', 'label': 'Show Pulse'},
                {'title': '', 'name': 'show_trace_bool', 'type': 'bool_push', 'label': 'Show Trace'},
            ] + Simulator.params}, ] + \
        [{'title': 'Spectrometer settings:', 'name': 'spectro_settings', 'type': 'group', 'children': [
             {'title': 'Min Wavelength (nm):', 'name': 'wl_min', 'type': 'float', 'value': 250,
              'tip': 'Minimal Wavelength of the virtual spectrometer'},
             {'title': 'Max Wavelength (nm):', 'name': 'wl_max', 'type': 'float', 'value': 1000,
              'tip': 'Minimal Wavelength of the virtual spectrometer'},

             {'title': 'Npoints:', 'name': 'npoints_spectro', 'type': 'list',
              'values': [2 ** n for n in range(8, 16)],
              'value': 512,
              'tip': 'Number of points of the spectrometer'},
        ]},
         {'title': 'Parameter value:', 'name': 'param_val', 'type': 'float', 'value': 0,
          'tip': 'Particular value at which to compute the NonLinear response and emit the spectrum'},
         ]

    def __init__(self, parent=None, params_state=None):
        super().__init__(parent, params_state)

        self.x_axis = None
        self.pulse = None
        self.scanned_axis = None
        self.ft = None
        self.max_pnps = None
        self.spectro_wavelength = None

    def commit_settings(self, param):
        """
        """

        if param.name() == 'show_pulse_bool':
            self.controller.update_pulse()

            intensity_t = lib.abs2(self.controller.pulse.field)
            intensity_t = intensity_t / np.max(intensity_t)
            phaset = np.unwrap(np.angle(self.controller.pulse.field))
            phaset -= lib.mean(phaset, intensity_t**2)

            intensity_w = lib.abs2(self.controller.pulse.spectrum)
            intensity_w = intensity_w / np.max(intensity_w)
            phasew = np.unwrap(np.angle(self.controller.pulse.spectrum))
            phasew -= lib.mean(phasew, intensity_w**2)

            self.data_grabed_signal_temp.emit([
                DataFromPlugins(name='FemtoPulse Temporal',
                                data=[intensity_t, phaset],
                                dim='Data1D', labels=['Intensity', 'Phase'],
                                x_axis=Axis(data=self.controller.pulse.t * 1e15, label='time', units='fs')),
                DataFromPlugins(name='FemtoPulse Spectral',
                                data=[intensity_w, phasew],
                                dim='Data1D', labels=['Amplitude', 'Phase'],
                                x_axis=Axis(data=convert(self.controller.pulse.w + self.controller.pulse.w0, "om", "wl"),
                                            label='Wavelength', units='nm'))
            ])
        elif param.name() == 'show_trace_bool':
            self.controller.update_pnps()
            self.data_grabed_signal_temp.emit([
                DataFromPlugins(name='Full Trace',
                                data=[normalize(np.fliplr(self.controller.trace.data.T) / self.controller.max_pnps)],
                                dim='Data2D', labels=['NL trace'],
                                x_axis=Axis(data=self.controller.trace.axes[0][::-1] * 1e15,
                                            label=self.controller.trace.labels[0], units='fs'),
                                y_axis=Axis(data=self.controller.trace.axes[1],
                                            label=self.controller.trace.labels[1], units='Hz')
                                ),])
            
        elif param.name() in iter_children(self.settings.child('spectro_settings'), []):
            self.update_spectro()
            
        elif param.name() in iter_children(self.settings.child('simul_settings'), []):
            self.controller.settings.child(*get_param_path(param)[3:]).setValue(param.value())


    def update_spectro(self):
        lambdamin = self.settings.child('spectro_settings', 'wl_min').value()
        lambdamax = self.settings.child('spectro_settings', 'wl_max').value()
        N = self.settings.child('spectro_settings', 'npoints_spectro').value()
        
        self.spectro_wavelength = np.linspace(lambdamin, lambdamax, N, endpoint=True)

    def get_scanned_axis(self):
        if self.controller is None:
            raise ValueError('The parametrized nonlinear process has not been defined')
        self.scanned_axis = Axis(data=self.controller.parameter, label=self.controller.trace.labels[1], units='')

        return self.scanned_axis

    def get_measure_axis(self):
        if self.controller is None:
            raise ValueError('The parametrized nonlinear process has not been defined')
        self.x_axis = Axis(data=self.controller.trace.axes[0], label=self.controller.trace.labels[0], units='')

        return self.x_axis

    def ini_detector(self, controller=None):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object) custom object of a PyMoDAQ plugin (Slave case). None if only one detector by controller (Master case)

        Returns
        -------
        self.status (edict): with initialization status: three fields:
            * info (str)
            * controller (object) initialized controller
            *initialized: (bool): False if initialization failed otherwise True
        """

        try:
            self.status.update(edict(initialized=False,info="",x_axis=None,y_axis=None,controller=None))
            if self.settings.child(('controller_status')).value() == "Slave":
                if controller is None:
                    raise Exception('no controller has been defined externally while this detector is a slave one')
                else:
                    self.controller = controller
            else:
                self.controller = Simulator(show_ui=False)
                #####################################

            ## TODO for your custom plugin
            # get the x_axis (you may want to to this also in the commit settings if x_axis may have changed
            self.get_measure_axis()
            self.emit_x_axis()
            self.update_spectro()

            ##############################

            self.status.info = "Your Python for pulse retrieval PyMoDAQ plugin is ready"
            self.status.initialized = True
            self.status.controller = self.controller
            return self.status

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [getLineInfo() + str(e), 'log']))
            self.status.info = getLineInfo() + str(e)
            self.status.initialized = False
            return self.status

    def close(self):
        """
        Terminate the communication protocol
        """
        pass
        ##

    def grab_data(self, Naverage=1, **kwargs):
        """

        Parameters
        ----------
        Naverage: (int) Number of hardware averaging
        kwargs: (dict) of others optionals arguments
        """
        if 'positions' in kwargs:
            parameter = kwargs['positions'][0] * 1e-15
        else:
            parameter = self.settings.child('param_val').value()
        self.controller.pnps.calculate(self.controller.pulse.spectrum, parameter)
        data = np.interp(self.spectro_wavelength,
                         np.flip(self.controller.pnps.process_wl * 1e9),
                         np.flip(self.controller.pnps.Tmn / self.controller.max_pnps))

        self.data_grabed_signal.emit([
            DataFromPlugins(name='PNPS',
                            data=[data],
                            dim='Data1D', labels=['NL trace'],
                            x_axis=Axis(data=self.spectro_wavelength,
                                        label='Wavelength', units='nm')
                            ), ])



    def stop(self):

        pass

        return ''
