#!/usr/bin/python
import os
import subprocess
import threading
import time

# Class for load, store, remove engine
class load_store_engine():
    def __init__(self, model_path, model_name, batch_size_gpu, batch_size_dla, num_devices, precision, ws_gpu, ws_dla, model_input, model_output ):
        self.model_path = model_path # Directory
        self.model_name = model_name # Model Name
        self.num_devices = num_devices # 3 if GPU+2DLA, 1 if GPU Only
        self.precision = precision # float16 or int8
        self.batch_size_gpu = batch_size_gpu # Batch Size for GPU
        self.batch_size_dla = batch_size_dla # Batch Size for DLA
        self.ws_gpu = ws_gpu # Workspace required for GPU
        self.ws_dla =ws_dla  # Workspace required for DLA
        self. model_input = model_input # Input name of the model
        self.model_output = model_output # Output name of the model
        self.trt_process = []

    def engine_gen(self):
        cmd = []
        model = []
        self.framework = os.path.splitext(self.model_name)[1]
        precision_cmd = str('--' + str(self.precision))
        in_io_format = str('--inputIOFormats=' + str(self.precision) + ':chw+chw4+chw32')
        for device_id in range(0, self.num_devices):
            if device_id == 1 or device_id == 2:
                self.device = 'dla'
                model_base_path = self._model2deploy()
                dla_cmd = str('--useDLACore=' + str(device_id - 1))
                workspace_cmd = str('--workspace=' + str(self.ws_dla))
                mempool_cmd = f'--memPoolSize=workspace:{self.ws_dla}'
                _model = str(os.path.splitext(self.model_name)[0]) + '_b' + str(self.batch_size_dla)+'_ws'+str(self.ws_dla) + '_' + str(self.device) + str(device_id)
                engine_CMD = str(
                    './trtexec' + " " + model_base_path + " " + in_io_format + " " +'--allowGPUFallback'+ " " + precision_cmd + " " + " " + dla_cmd + " " +
                    workspace_cmd)
            else:
                self.device = 'gpu'
                model_base_path = self._model2deploy()
                workspace_cmd = str('--workspace=' + str(self.ws_gpu))
                mempool_cmd = f'--memPoolSize=workspace:{self.ws_gpu}'
                _model = str(os.path.splitext(self.model_name)[0]) + '_b' + str(self.batch_size_gpu) + '_ws' + str(
                    self.ws_gpu) + '_' + str(self.device)
                # engine_CMD = str(
                #     './trtexec' + " " + model_base_path + " " + in_io_format + " " + precision_cmd + " " +workspace_cmd)
                engine_CMD = f'./trtexec {model_base_path} {in_io_format} {precision_cmd} {mempool_cmd}'
            cmd.append(engine_CMD)
            model.append(_model)
            
        return cmd, model

    def check_downloaded_models(self, model_name, framework):
        model_files = []
        if framework == str('onnx'):
            model_name_split = os.path.splitext(model_name)[0]
            model_files.append(str(model_name_split + '-bs' + str(self.batch_size_gpu) + '.' + framework))
            if self.num_devices > 1:
                model_files.append(str(model_name_split + '-bs' + str(self.batch_size_dla) + '.' + framework))
        else:
            model_files.append(model_name)

        for e_id in range(0, len(model_files)):
            model_file = os.path.join(self.model_path, model_files[e_id])
            if not os.path.isfile(model_file):
                print('Could Not find model file {} in {}\nPlease Download all model files'.format(model_files[e_id], self.model_path))
                return True
        return False

    def _model2deploy(self):
        if self.framework == str('.prototxt'):
            model_full_path = os.path.join(self.model_path, self.model_name)
            model_full_path = os.path.abspath(model_full_path)  # Convert to absolute path
            _model_output = ''
            # _out_io_format = '--outputIOFormats='
            out_names = self.model_output.split(":")
            for out in out_names:
                _model_output += str('--output=' + str(out) + ' ')
            _model_output = ''
            out_io_formats = []
            out_names = self.model_output.split(":")
            for out in out_names:
                _model_output += f'--output={out} '
                out_io_formats.append(f'{self.precision}:chw+chw4+chw32')
            _out_io_format = '--outputIOFormats=' + ','.join(out_io_formats)
            # for idx in range(len(out_names)):
            #     _out_io_format += str(str(self.precision) + ':chw+chw4+chw32,')
            
            #_model_output = str('--output=' + str(self.model_output))
            # _model_base = str('--deploy=' + str(os.path.join(self.model_path, self.model_name)))
            _model_base = f'--deploy={model_full_path}'
            if self.device=='gpu':
                batch_cmd = str('--batch=' + str(self.batch_size_gpu))
            elif self.device == 'dla':
                batch_cmd = str('--batch=' + str(self.batch_size_dla))
            return str(_model_output + " " + _out_io_format + " " + _model_base+ " " + batch_cmd)
        if self.framework == str('.onnx'):
            batch_cmd = str('--explicitBatch')
            model_name_split = os.path.splitext(self.model_name)[0]
            if self.device == 'gpu':
                model_onnx = str(model_name_split+'-bs'+str(self.batch_size_gpu)+self.framework)
            if self.device == 'dla':
                model_onnx = str(model_name_split+'-bs'+str(self.batch_size_dla)+self.framework)
            return str('--onnx=' + str(os.path.join(self.model_path, model_onnx))+ " " + batch_cmd)
        if self.framework == str('.uff'):
            _model_input = str('--uffInput='+str(self.model_input))
            _model_output = str('--output='+str(self.model_output))
            _model_base = str('--uff=' + str(os.path.join(self.model_path, self.model_name)))
            if self.device == 'gpu':
                batch_cmd = str('--batch=' + str(self.batch_size_gpu))
            elif self.device == 'dla':
                batch_cmd = str('--batch=' + str(self.batch_size_dla))
            return  str(_model_input+" "+_model_output+" "+_model_base+ " " + batch_cmd)


    # def save_engine(self, _cmds, _models):
    #     save_engine_path = str('--saveEngine=' + str(os.path.join(self.model_path, _models)) + '.engine')
    #     cmd = str(_cmds)+" "+str(save_engine_path)
    #     trt_process = subprocess.Popen([cmd], cwd='/usr/src/tensorrt/bin/', shell=True, stdout=subprocess.DEVNULL,
    #                                    stderr=subprocess.STDOUT)
    #     while trt_process.poll() == None:
    #         trt_process.poll()
    #     trt_process.kill()
    def save_engine(self, _cmds, _models):
        save_engine_path = '--saveEngine=' + os.path.join(self.model_path, _models) + '.engine'
        cmd = _cmds + " " + save_engine_path
        trt_process = subprocess.Popen(cmd, cwd='/usr/src/tensorrt/bin/', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = trt_process.communicate()  # Wait for the process to complete
    
        if trt_process.returncode != 0:
            print(f"Error building engine for {_models}:")
            print(stderr.decode('utf-8'))
        else:
            print(f"Successfully built engine for {_models}.")

    def save_all(self, commands, models):
        for e_id in range(0, self.num_devices):
            self.save_engine(commands[e_id], models[e_id])
            
    def load_engine(self, _cmds, _models, load_output):
        load_engine_path = '--loadEngine=' + os.path.join(self.model_path, _models) + '.engine'
        avgruns_cmd = '--avgRuns=100 --duration=180'
        cmd = _cmds + " " + avgruns_cmd + " " + load_engine_path
        trt_process = subprocess.Popen(cmd, cwd='/usr/src/tensorrt/bin/', shell=True, stdout=load_output, stderr=load_output)
        trt_process.wait()  # Wait for the process to complete
        if trt_process.returncode != 0:
            print(f"Error loading engine for {_models}. Check {load_output.name} for details.")
        else:
            print(f"Successfully loaded engine for {_models}.")

    # def load_all(self, commands, models):
    #     load_threads = []
    #     load_file_list = []
    #     for e_id in range(0, self.num_devices):
    #         load_file = os.path.join(self.model_path, models[e_id] + '.txt')
    #         load_output = open(load_file, 'w')
    #         _load_threads = threading.Thread(target=self.load_engine(commands[e_id], models[e_id], load_output))
    #         load_threads.append(_load_threads)
    #         load_file_list.append(load_output)
    #         time.sleep(10)# Load memory
    #     # Start Threads 
    #     for lt in load_threads:
    #         lt.start()
    #     # Wait till threads are synchronize
    #     for lt in load_threads:
    #         lt.join()
    #     # Kill the subprocessess once complete
    #     for tp in self.trt_process:
    #         while tp.poll() == None:
    #             tp.poll()
    #         tp.kill()
    #     for flist in load_file_list:
    #         flist.close()
    def load_all(self, commands, models):
        load_threads = []
        load_file_list = []
        for e_id in range(0, self.num_devices):
            load_file = os.path.join(self.model_path, models[e_id] + '.txt')
            load_output = open(load_file, 'w')
            _load_thread = threading.Thread(target=self.load_engine, args=(commands[e_id], models[e_id], load_output))
            load_threads.append(_load_thread)
            load_file_list.append(load_output)
            time.sleep(10)  # Load memory
        # Start Threads
        for lt in load_threads:
            lt.start()
        # Wait till threads are complete
        for lt in load_threads:
            lt.join()
        # Close output files
        for flist in load_file_list:
            flist.close()

    def remove_engine(self, models):
        _engine_path = str(str(os.path.join(self.model_path, models)) + '.engine')
        _txtout_path = str(str(os.path.join(self.model_path, models)) + '.txt')
        if os.path.isfile(_engine_path):
            os.remove(_engine_path)
        if os.path.isfile(_txtout_path):
            os.remove(_txtout_path)

    def remove_all(self, models):
        for e_id in range(0, self.num_devices):
            self.remove_engine(models[e_id])
