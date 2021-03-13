import numpy as np
from easydict import EasyDict as edict
from pymodaq.daq_utils.daq_utils import ThreadCommand, getLineInfo, DataFromPlugins, Axis, gauss1D, normalize
from pymodaq.daq_utils.array_manipulation import crop_vector_to_axis, linspace_this_vect
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
                {'title': 'Show fund. spectrum:', 'name': 'show_pulse_bool', 'type': 'bool'},
                {'title': 'Show Trace:', 'name': 'show_trace_bool', 'type': 'bool'},
            ] + Simulator.params}, ] + \
        [{'title': 'Spectrometer settings:', 'name': 'spectro_settings', 'type': 'group', 'children': [
             {'title': 'Min Wavelength (nm):', 'name': 'wl_min', 'type': 'float', 'value': 250,
              'tip': 'Minimal Wavelength of the virtual spectrometer'},
             {'title': 'Max Wavelength (nm):', 'name': 'wl_max', 'type': 'float', 'value': 550,
              'tip': 'Minimal Wavelength of the virtual spectrometer'},

             {'title': 'Npoints:', 'name': 'npoints_spectro', 'type': 'list',
              'values': [2 ** n for n in range(8, 16)],
              'value': 512,
              'tip': 'Number of points of the spectrometer'},
        ]},
         {'title': 'Scanning device value:', 'name': 'param_val', 'type': 'float', 'value': 0,
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
        if param.name() == 'show_trace_bool':
            self.settings.child('simul_settings', 'show_pulse_bool').setValue(False)
        elif param.name() == 'show_pulse_bool':
            self.settings.child('simul_settings', 'show_trace_bool').setValue(False)

        elif param.name() in iter_children(self.settings.child('spectro_settings'), []):
            self.update_spectro()
            
        elif param.name() in iter_children(self.settings.child('simul_settings'), []):
            self.controller.settings.child(*get_param_path(param)[3:]).setValue(param.value())
            self.controller.update_pnps()


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
            self.controller.update_pnps()

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
        if self.settings.child('simul_settings', 'show_pulse_bool').value():
            self.controller.update_pulse()
            w = self.controller.pulse.w

            wl_lim = self.settings.child('spectro_settings', 'wl_min').value() * 1e-9, \
                     self.settings.child('spectro_settings', 'wl_max').value() * 1e-9
            wl = convert(w + self.controller.pulse.w0, "om", "wl")
            spectrum = normalize(self.controller.pulse.spectrum * wl**2)
            intensity_w = lib.abs2(spectrum)
            intensity_w = intensity_w / np.max(intensity_w)
            wlc, intensity_croped = crop_vector_to_axis(wl[::-1], intensity_w[::-1], wl_lim)
            wlc_lin, intensity_lin = linspace_this_vect(wlc, intensity_croped,
                                                       self.settings.child('spectro_settings',
                                                                           'npoints_spectro').value())

            self.data_grabed_signal.emit([
                DataFromPlugins(name='FemtoPulse Spectral',
                                data=[intensity_lin],
                                dim='Data1D', labels=['Spectral Intensity'],
                                x_axis=Axis(data=wlc_lin,
                                            label='Wavelength', units='m'))
            ])
        elif self.settings.child('simul_settings', 'show_trace_bool').value():
            self.controller.update_pnps()
            wl_lim = self.settings.child('spectro_settings', 'wl_min').value() * 1e-9, \
                     self.settings.child('spectro_settings', 'wl_max').value() * 1e-9
            Npts = self.settings.child('spectro_settings',
                                'npoints_spectro').value()
            data, axis, parameter_axis = self.controller.trace_exp(Npts=Npts, wl_lim=wl_lim)
            self.data_grabed_signal.emit([
                DataFromPlugins(name='Full Trace', data=[data.T], dim='Data2D', labels=['NL trace'],
                                x_axis=parameter_axis, y_axis=axis)])
        else:
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
