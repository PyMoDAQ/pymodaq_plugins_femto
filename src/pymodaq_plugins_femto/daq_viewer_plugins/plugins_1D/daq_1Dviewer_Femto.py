import numpy as np
from easydict import EasyDict as edict
from pymodaq.daq_utils.daq_utils import ThreadCommand, getLineInfo, DataFromPlugins, Axis, gauss1D, normalize
from pymodaq.daq_viewer.utility_classes import DAQ_Viewer_base, comon_parameters
from pathlib import Path
from pypret.frequencies import om2wl, wl2om, convert
from pypret import FourierTransform, Pulse, PNPS, PulsePlot, lib
from scipy.interpolate import interp2d

class DAQ_1DViewer_Femto(DAQ_Viewer_base):
    """
    """
    params = comon_parameters+[
        {'title': 'Algorithm Options:', 'name': 'algo', 'type': 'group', 'children': [
            {'title': 'Method:', 'name': 'method', 'type': 'list', 'values': ['frog', 'tdp', 'dscan', 'miips', 'ifrog'],
             'tip': 'Characterization Method'},
            {'title': 'NL process:', 'name': 'nlprocess', 'type': 'list', 'values': ['shg', 'thg', 'sd', 'pg', 'tg'],
             'tip': 'Non Linear process used in the experiment'},
        ]},
        {'title': 'Grid settings:', 'name': 'grid_settings', 'type': 'group', 'children': [
            {'title': 'Central Wavelength (nm):', 'name': 'wl0', 'type': 'float', 'value': 750,
             'tip': 'Central Wavelength of the Pulse spectrum and frequency grid'},
            {'title': 'Npoints:', 'name': 'npoints', 'type': 'list', 'values': [2**n for n in range(8,16)], 'value':512,
             'tip': 'Number of points for the temporal and Fourier Transform Grid'},
            {'title': 'Time resolution (fs):', 'name': 'time_resolution', 'type': 'float', 'value': 0.5,
             'tip': 'Time spacing between 2 points in the time grid'},
            ]},
        {'title': 'Pulse Source:', 'name': 'pulse_source', 'type': 'list', 'values': ['Simulated', 'From File'],},
        {'title': 'Show Pulse:', 'name': 'show_pulse', 'type': 'bool_push', 'value': False, },
        {'title': 'Show trace:', 'name': 'show_trace', 'type': 'bool_push', 'value': False, },
        {'title': 'Pulse Settings:', 'name': 'pulse_settings', 'type': 'group', 'children': [
            {'title': 'FWHM (fs):', 'name': 'fwhm_time', 'type': 'float', 'value': 5,
             'tip': 'Fourier Limited Pulse duration in femtoseconds'},
            {'title': 'GDD (fs2):', 'name': 'GDD_time', 'type': 'float', 'value': 245,
             'tip': 'Group Delay Dispersion in femtosecond square'},
            {'title': 'TOD (fs3):', 'name': 'TOD_time', 'type': 'float', 'value': 100,
             'tip': 'Third Order Dispersion in femtosecond cube'},
            {'title': 'Data File:', 'name': 'data_file_path', 'type': 'browsepath', 'filetype': True, 'visible': False,
             'value': str(Path(__file__).parent.parent.parent.joinpath('data/spectral_data.csv')),
             'tip': 'Path to a CSV file containing in columns: wavelength(nm), Normalized Sprectral Intensity and phase'
                    ' in radians'},

        ]},
        ]

    def __init__(self, parent=None, params_state=None):
        super().__init__(parent, params_state)

        self.x_axis = None
        self.pulse = None
        self.scanned_axis = None
        self.ft = None
        self.max_pnps = None

    def commit_settings(self, param):
        """
        """
        if param.name() == 'pulse_source':
            for child in self.settings.child('pulse_settings').children():
                if child.name() == 'data_file_path':
                    child.show(param.value() == 'From File')
                else:
                    child.show(param.value() != 'From File')

        elif param.name() == 'show_pulse':
            self.update_pulse()

            intensity_t = lib.abs2(self.pulse.field)
            intensity_w = intensity_t / np.max(intensity_t)
            phaset = np.unwrap(np.angle(self.pulse.field))
            phaset -= lib.mean(phaset, intensity_t**2)

            intensity_w = lib.abs2(self.pulse.spectrum)
            intensity_w = intensity_w / np.max(intensity_w)
            phasew = np.unwrap(np.angle(self.pulse.spectrum))
            phasew -= lib.mean(phasew, intensity_w**2)

            self.data_grabed_signal_temp.emit([
                DataFromPlugins(name='FemtoPulse Temporal',
                                data=[intensity_t, phaset],
                                dim='Data1D', labels=['Intensity', 'Phase'],
                                x_axis=Axis(data=self.pulse.t * 1e15, label='time', units='fs')),
                DataFromPlugins(name='FemtoPulse Spectral',
                                data=[intensity_w, phasew],
                                dim='Data1D', labels=['Amplitude', 'Phase'],
                                x_axis=Axis(data=convert(self.pulse.w + self.pulse.w0, "om", "wl"),
                                            label='Wavelength', units='nm'))
            ])
        elif param.name() == 'show_trace':
            self.update_pnps()
            self.data_grabed_signal_temp.emit([
                DataFromPlugins(name='Full Trace',
                                data=[normalize(self.controller.trace.data / self.max_pnps)],
                                dim='Data2D', labels=['NL trace'],
                                x_axis=Axis(data=self.controller.trace.axes[0] * 1e15,
                                            label=self.controller.trace.labels[0], units='fs'),
                                y_axis=Axis(data=self.controller.trace.axes[1],
                                            label=self.controller.trace.labels[1], units='Hz')
                                ),])

    def update_grid(self):
        Nt = self.settings.child('grid_settings', 'npoints').value()
        dt = self.settings.child('grid_settings', 'time_resolution').value() * 1e-15
        wl0 = self.settings.child('grid_settings', 'wl0').value() * 1e-9
        self.ft = FourierTransform(Nt, dt=dt, w0=wl2om(-wl0-300e-9))

    def update_pnps(self):

        pulse = self.update_pulse()
        method = self.settings.child('algo', 'method').value()
        process = self.settings.child('algo', 'nlprocess').value()
        self.controller = PNPS(pulse, method, process)
        parameter = np.linspace(self.ft.t[-1], self.ft.t[0], len(self.ft.t))
        self.controller.calculate(pulse.spectrum, parameter)
        self.max_pnps = np.max(self.controller.Tmn)
        return self.controller

    def update_pulse(self):
        self.update_grid()
        wl0 = self.settings.child('grid_settings', 'wl0').value() * 1e-9
        pulse = Pulse(self.ft, wl0)

        if self.settings.child('pulse_source').value() == 'Simulated':
            fwhm = self.settings.child('pulse_settings', 'fwhm_time').value()
            GDD = self.settings.child('pulse_settings', 'GDD_time').value()
            TOD = self.settings.child('pulse_settings', 'TOD_time').value()

            pulse.field = gauss1D(pulse.t, x0=0, dx=0.5 * (fwhm * 1e-15) / np.sqrt(np.log(2.0)))
            pulse.spectrum = (pulse.spectrum) * np.exp(
                1j * (GDD * 1e-30) * ((pulse.w - pulse.w0) ** 2) / 2 + 1j * (TOD * 1e-45) * (
                            (pulse.w - pulse.w0) ** 3) / 6)

            # recenter pulse in time domain
            idx = np.argmax(pulse.intensity)
            pulse.spectrum = pulse.spectrum * np.exp(-1j * pulse.t[idx] * (pulse.w - pulse.w0))

        else:
            data_path = self.settings.child('pulse_settings', 'data_file_path').value()
            data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
            in_wl, in_int, in_phase = (data[:, i] for i in range(3))

            in_int = np.interp(pulse.wl, in_wl * 1e-9, np.maximum(0, in_int), left=0, right=0)
            in_phase = np.interp(pulse.wl, in_wl * 1e-9, in_phase, left=0, right=0)
            pulse.spectrum = in_int * np.exp(1j * in_phase)

        self.pulse = pulse
        return pulse

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
                ## TODO for your custom plugin
                self.controller = self.update_pnps()
                #####################################

            ## TODO for your custom plugin
            # get the x_axis (you may want to to this also in the commit settings if x_axis may have changed
            self.get_measure_axis()
            self.emit_x_axis()

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
            parameter = 0.
        self.controller.calculate(self.pulse.spectrum, parameter)

        self.data_grabed_signal.emit([
            DataFromPlugins(name='PNPS',
                            data=[self.controller.Tmn / self.max_pnps],
                            dim='Data1D', labels=['NL trace'],
                            x_axis=Axis(data=self.controller.process_wl*1e9*2,
                                        label='Wavelength', units='nm')
                            ), ])



    def stop(self):

        pass

        return ''
