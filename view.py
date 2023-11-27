import customtkinter as CTk

from calculate import calculate


class App(CTk.CTk):
    def __init__(self):
        super().__init__()

        self.geometry('800x1200')
        self.title('Geostochastic generator with NNC')
        self.header_font = CTk.CTkFont(weight="bold", size=20)


        #Covar Frame
        self.covar_frame = CTk.CTkFrame(master=self, fg_color='transparent')
        self.covar_frame.grid(row=0, column=0, padx=(20, 20), sticky='nsew')
        self.covar_label = CTk.CTkLabel(master=self.covar_frame,
                                        text='Covariance parameters',
                                        fg_color='transparent',
                                        font=self.header_font)
        self.covar_label.grid(row=0, column=0)

        #Model
        self.model_label = CTk.CTkLabel(master=self.covar_frame,
                                        text='Model',
                                        fg_color='transparent')
        self.model_label.grid(row=1, column=0)
        self.models_list = ['gaussian', 'exponential', 'spherical', 'hyperbolic']
        self.model_combobox = CTk.CTkComboBox(master=self.covar_frame, values=self.models_list)
        self.model_combobox.grid(row=1, column=1)
        self.dim_label = CTk.CTkLabel(master=self.covar_frame,
                                        text='Num dim',
                                        fg_color='transparent')
        self.dim_label.grid(row=1, column=2)
        self.dim_list = ['1', '2', '3']
        self.dim_combobox = CTk.CTkComboBox(master=self.covar_frame, values=self.dim_list)
        self.dim_combobox.grid(row=1, column=3)

        #Range
        self.range0_width = 50
        self.range0_label = CTk.CTkLabel(master=self.covar_frame,
                                        text='Initial range',
                                        fg_color='transparent')
        self.range0_label.grid(row=3, column=0)
        self.range0_label_x = CTk.CTkLabel(master=self.covar_frame,
                                        text='X',
                                        fg_color='transparent', width=self.range0_width)
        self.range0_label_x.grid(row=2, column=1)
        self.range0_entry_x = CTk.CTkEntry(master=self.covar_frame, width=self.range0_width)
        self.range0_entry_x.grid(row=3, column=1)
        self.range0_label_y = CTk.CTkLabel(master=self.covar_frame,
                                           text='Y',
                                           fg_color='transparent', width=self.range0_width)
        self.range0_label_y.grid(row=2, column=2)
        self.range0_entry_y = CTk.CTkEntry(master=self.covar_frame, width=self.range0_width)
        self.range0_entry_y.grid(row=3, column=2)
        self.range0_label_z = CTk.CTkLabel(master=self.covar_frame,
                                           text='Z',
                                           fg_color='transparent', width=self.range0_width)
        self.range0_label_z.grid(row=2, column=3)
        self.range0_entry_z = CTk.CTkEntry(master=self.covar_frame, width=self.range0_width)
        self.range0_entry_z.grid(row=3, column=3)

        #Azimuth
        self.azimuth_width = 50
        self.azimuth_label = CTk.CTkLabel(master=self.covar_frame,
                                         text='Azimuth',
                                         fg_color='transparent')
        self.azimuth_label.grid(row=5, column=0)
        self.azimuth_label_x = CTk.CTkLabel(master=self.covar_frame,
                                           text='X/Y',
                                           fg_color='transparent', width=self.azimuth_width)
        self.azimuth_label_x.grid(row=4, column=1)
        self.azimuth_entry_x = CTk.CTkEntry(master=self.covar_frame, width=self.azimuth_width)
        self.azimuth_entry_x.grid(row=5, column=1)
        self.azimuth_label_y = CTk.CTkLabel(master=self.covar_frame,
                                           text='Y/Z',
                                           fg_color='transparent', width=self.azimuth_width)
        self.azimuth_label_y.grid(row=4, column=2)
        self.azimuth_entry_y = CTk.CTkEntry(master=self.covar_frame, width=self.azimuth_width)
        self.azimuth_entry_y.grid(row=5, column=2)
        self.azimuth_label_z = CTk.CTkLabel(master=self.covar_frame,
                                           text='X/Z',
                                           fg_color='transparent', width=self.azimuth_width)
        self.azimuth_label_z.grid(row=4, column=3)
        self.azimuth_entry_z = CTk.CTkEntry(master=self.covar_frame, width=self.azimuth_width)
        self.azimuth_entry_z.grid(row=5, column=3)

        #Sill
        self.sill_width = 50
        self.c0_label = CTk.CTkLabel(master=self.covar_frame,
                                     text='Sill',
                                     fg_color='transparent')
        self.c0_label.grid(row=6, column=0)
        self.c0_entry = CTk.CTkEntry(master=self.covar_frame, width=self.sill_width)
        self.c0_entry.grid(row=6, column=1)

        #Alpha
        self.alpha_width = 50
        self.alpha_label = CTk.CTkLabel(master=self.covar_frame,
                                     text='Alpha parameter',
                                     fg_color='transparent')
        self.alpha_label.grid(row=7, column=0)
        self.alpha_entry = CTk.CTkEntry(master=self.covar_frame, width=self.alpha_width)
        self.alpha_entry.grid(row=7, column=1)

        #Nugget
        self.nugget_width = 50
        self.nugget_label = CTk.CTkLabel(master=self.covar_frame,
                                     text='Nugget',
                                     fg_color='transparent')
        self.nugget_label.grid(row=8, column=0)
        self.nugget_entry = CTk.CTkEntry(master=self.covar_frame, width=self.nugget_width)
        self.nugget_entry.grid(row=8, column=1)


        #Neigh Frame
        self.neigh_frame = CTk.CTkFrame(master=self, fg_color='transparent')
        self.neigh_frame.grid(row=1, column=0, padx=(20, 20), sticky='nsew')
        self.neigh_label = CTk.CTkLabel(master=self.neigh_frame,
                                        text='Neighborhood parameters',
                                        fg_color='transparent',
                                        font=self.header_font)
        self.neigh_label.grid(row=0, column=0)

        #Wradius
        self.wradius_width = 50
        self.wradius_label = CTk.CTkLabel(master=self.neigh_frame,
                                     text='Search radius',
                                     fg_color='transparent')
        self.wradius_label.grid(row=1, column=0)
        self.wradius_entry = CTk.CTkEntry(master=self.neigh_frame, width=self.wradius_width)
        self.wradius_entry.grid(row=1, column=1)

        #Lookup
        self.lookup_label = CTk.CTkLabel(master=self.neigh_frame,
                                     text='Lookup table',
                                     fg_color='transparent')
        self.lookup_label.grid(row=2, column=0)
        self.lookup_checkbox = CTk.CTkCheckBox(master=self.neigh_frame, text='')
        self.lookup_checkbox.grid(row=2, column=1)

        #Nb
        self.nb_width = 50
        self.nb_label = CTk.CTkLabel(master=self.neigh_frame,
                                     text='Number of neighbors',
                                     fg_color='transparent')
        self.nb_label.grid(row=3, column=0)
        self.nb_entry = CTk.CTkEntry(master=self.neigh_frame, width=self.nb_width)
        self.nb_entry.grid(row=3, column=1)

        #Params Frame
        self.params_frame = CTk.CTkFrame(master=self, fg_color='transparent')
        self.params_frame.grid(row=2, column=0, padx=(20, 20), sticky='nsew')
        self.params_label = CTk.CTkLabel(master=self.params_frame,
                                        text='Mesh and Calculation parameters',
                                        fg_color='transparent',
                                        font=self.header_font)
        self.params_label.grid(row=0, column=0)

        #Grid
        self.grid_width = 50
        self.grid_label = CTk.CTkLabel(master=self.params_frame,
                                     text='Grid',
                                     fg_color='transparent')
        self.grid_label.grid(row=2, column=0)
        self.grid_label_x = CTk.CTkLabel(master=self.params_frame,
                                     text='X',
                                     fg_color='transparent')
        self.grid_label_x.grid(row=1, column=1)
        self.grid_label_y = CTk.CTkLabel(master=self.params_frame,
                                     text='Y',
                                     fg_color='transparent')
        self.grid_label_y.grid(row=1, column=2)
        self.grid_label_z = CTk.CTkLabel(master=self.params_frame,
                                         text='Z',
                                         fg_color='transparent')
        self.grid_label_z.grid(row=1, column=3)
        self.grid_entry_x = CTk.CTkEntry(master=self.params_frame, width=self.grid_width)
        self.grid_entry_x.grid(row=2, column=1)
        self.grid_entry_y = CTk.CTkEntry(master=self.params_frame, width=self.grid_width)
        self.grid_entry_y.grid(row=2, column=2)
        self.grid_entry_z = CTk.CTkEntry(master=self.params_frame, width=self.grid_width)
        self.grid_entry_z.grid(row=2, column=3)

        #M
        self.m_width = 50
        self.m_label = CTk.CTkLabel(master=self.params_frame,
                                     text='Number of realizations',
                                     fg_color='transparent')
        self.m_label.grid(row=3, column=0)
        self.m_entry = CTk.CTkEntry(master=self.params_frame, width=self.m_width)
        self.m_entry.grid(row=3, column=1)

        #Mean
        self.mean_width = 50
        self.mean_label = CTk.CTkLabel(master=self.params_frame,
                                    text='Mean',
                                    fg_color='transparent')
        self.mean_label.grid(row=4, column=0)
        self.mean_entry = CTk.CTkEntry(master=self.params_frame, width=self.mean_width)
        self.mean_entry.grid(row=4, column=1)

        #NNC
        self.nnc_label = CTk.CTkLabel(master=self.params_frame,
                                         text='NNC',
                                         fg_color='transparent')
        self.nnc_label.grid(row=5, column=0)
        self.nnc_checkbox = CTk.CTkCheckBox(master=self.params_frame, text='')
        self.nnc_checkbox.grid(row=5, column=1)

        #Category
        self.cat_label = CTk.CTkLabel(master=self.params_frame,
                                         text='Category',
                                         fg_color='transparent')
        self.cat_label.grid(row=6, column=0)
        self.cat_checkbox = CTk.CTkCheckBox(master=self.params_frame, text='')
        self.cat_checkbox.grid(row=6, column=1)
        self.cat_threshold_width = 75
        self.cat_threshold_entry = CTk.CTkEntry(master=self.params_frame, width=self.cat_threshold_width,
                                                placeholder_text='Threshold')
        self.cat_threshold_entry.grid(row=6, column=2)

        #Debug
        self.debug_label = CTk.CTkLabel(master=self.params_frame,
                                         text='Debug',
                                         fg_color='transparent')
        self.debug_label.grid(row=7, column=0)

        #Compare
        self.debug_checkbox = CTk.CTkCheckBox(master=self.params_frame, text='Compare')
        self.debug_checkbox.grid(row=7, column=1)

        #Frob
        self.frob_checkbox = CTk.CTkCheckBox(master=self.params_frame, text='Calculation Frobenius norm')
        self.frob_checkbox.grid(row=7, column=2)

        #Seeds
        self.seed_width = 75
        self.seed_search_entry = CTk.CTkEntry(master=self.params_frame, width=self.seed_width,
                                                placeholder_text='Search seed')
        self.seed_search_entry.grid(row=8, column=1)
        self.seed_path_entry = CTk.CTkEntry(master=self.params_frame, width=self.seed_width,
                                              placeholder_text='Path seed')
        self.seed_path_entry.grid(row=8, column=2)
        self.seed_U_entry = CTk.CTkEntry(master=self.params_frame, width=self.seed_width,
                                              placeholder_text='U seed')
        self.seed_U_entry.grid(row=8, column=3)
        self.seed_label = CTk.CTkLabel(master=self.params_frame,
                                         text='Leave for random',
                                         fg_color='transparent')
        self.seed_label.grid(row=8, column=4)



        #Plot Frame
        self.plot_frame = CTk.CTkFrame(master=self, fg_color='transparent')
        self.plot_frame.grid(row=3, column=0, padx=(20, 20), sticky='nsew')
        self.plot_label = CTk.CTkLabel(master=self.plot_frame,
                                         text='Plot parameters',
                                         fg_color='transparent',
                                         font=self.header_font)
        self.plot_label.grid(row=0, column=0)

        #Mode
        self.mode_label = CTk.CTkLabel(master=self.plot_frame,
                                         text='Mode',
                                         fg_color='transparent')
        self.mode_label.grid(row=1, column=0)
        self.mode_list = ['histo', 'vario', 'qq', 'all']
        self.mode_combobox = CTk.CTkComboBox(master=self.plot_frame, values=self.mode_list)
        self.mode_combobox.grid(row=1, column=1)
        self.vario_width = 80
        self.cutoff_entry = CTk.CTkEntry(master=self.plot_frame, width=self.vario_width,
                                         placeholder_text='Vario Cutoff')
        self.cutoff_entry.grid(row=1, column=2)
        self.bins_entry = CTk.CTkEntry(master=self.plot_frame, width=self.vario_width,
                                       placeholder_text='Vario Bins')
        self.bins_entry.grid(row=1, column=3)

        #Out format
        self.out_width = 120
        self.screen = CTk.CTkCheckBox(master=self.plot_frame, text='Plot on screen')
        self.screen.grid(row=2, column=0)
        self.save = CTk.CTkCheckBox(master=self.plot_frame, text='Save')
        self.save.grid(row=2, column=1)
        self.savefilename_entry = CTk.CTkEntry(master=self.plot_frame, width=self.out_width,
                                       placeholder_text='Leave for random')
        self.savefilename_entry.grid(row=2, column=2)

        #Show_nnc
        self.show_nnc = CTk.CTkCheckBox(master=self.plot_frame, text='Show NNC')
        self.show_nnc.grid(row=3, column=0)

        #Config Parameters
        self.config_frame = CTk.CTkFrame(master=self, fg_color='transparent')
        self.config_frame.grid(row=4, column=0, padx=(20, 20), sticky='nsew')
        self.config_label = CTk.CTkLabel(master=self.config_frame,
                                       text='Config parameters',
                                       fg_color='transparent',
                                       font=self.header_font)
        self.config_label.grid(row=0, column=0)

        #Config filename
        self.config_filename_width = 100
        self.config_filename_label = CTk.CTkLabel(master=self.config_frame,
                                       text='Config filename',
                                       fg_color='transparent')
        self.config_filename_label.grid(row=1, column=0)
        self.config_filename = CTk.CTkEntry(master=self.config_frame, width=self.config_filename_width,
                                       placeholder_text='config')
        self.config_filename.grid(row=1, column=1)

        #Buttons Frame
        self.buttons_frame = CTk.CTkFrame(master=self, fg_color='transparent')
        self.buttons_frame.grid(row=5, column=0, padx=(20, 20), sticky='nsew')

        #Generate button
        self.btn_generate = CTk.CTkButton(master=self.buttons_frame, text='Write to config', width=100,
                                          command=self._write_config)
        self.btn_generate.grid(row=0, column=0)

        #Calculate button
        self.btn_calculate = CTk.CTkButton(master=self.buttons_frame, text='Calculate', width=100,
                                           command=self._calculate)
        self.btn_calculate.grid(row=0, column=1)

        #Read button
        self.btn_read = CTk.CTkButton(master=self.buttons_frame, text='Read config', width=100,
                                          command=self._read_config)
        self.btn_read.grid(row=0, column=2)

    def _write_config(self):
        with open(self._get_config_filename(), 'w') as f:
            f.write(f'model {self.model_combobox.get()}\n')
            f.write(f'ndim {self.dim_combobox.get()}\n')
            range0_str = 'range ' + f'{self.range0_entry_x.get()}'
            if len(self.range0_entry_y.get()) != 0:
                range0_str += f' {self.range0_entry_y.get()}'
            if len(self.range0_entry_z.get()) != 0:
                range0_str += f' {self.range0_entry_z.get()}'
            f.write(range0_str+'\n')

            azimuth_str = f'azimuth {self.azimuth_entry_x.get()}'
            if len(self.azimuth_entry_y.get()) != 0:
                azimuth_str += f' {self.azimuth_entry_y.get()}'
            if len(self.azimuth_entry_z.get()) != 0:
                azimuth_str += f' {self.azimuth_entry_z.get()}'
            f.write(azimuth_str+'\n')

            f.write(f'c0 {self.c0_entry.get()}\n')
            f.write(f'alpha {self.alpha_entry.get()}\n')
            f.write(f'nugget {self.nugget_entry.get()}\n')
            f.write(f'wradius {self.wradius_entry.get()}\n')
            f.write(f'lookup {self.lookup_checkbox.get()}\n')
            f.write(f'nb {self.nb_entry.get()}\n')
            f.write(f'nx {self.grid_entry_x.get()}\n')
            f.write(f'ny {self.grid_entry_y.get()}\n')
            f.write(f'nz {self.grid_entry_z.get()}\n')
            f.write(f'm {self.m_entry.get()}\n')
            f.write(f'mean {self.mean_entry.get()}\n')
            f.write(f'nnc {self.nnc_checkbox.get()}\n')
            f.write(f'category {self.cat_checkbox.get()}\n')
            f.write(f'cat_threshold {self.cat_threshold_entry.get()}\n')
            f.write(f'debug {self.debug_checkbox.get()}\n')
            f.write(f'calc_frob {self.frob_checkbox.get()}\n')
            f.write(f'seed_search {self.seed_search_entry.get()}\n')
            f.write(f'seed_path {self.seed_path_entry.get()}\n')
            f.write(f'seed_U {self.seed_U_entry.get()}\n')
            f.write(f'cutoff {self.cutoff_entry.get()}\n')
            f.write(f'bins {self.bins_entry.get()}\n')
            f.write(f'show {self.screen.get()}\n')
            f.write(f'save {self.save.get()}\n')
            if len(self.savefilename_entry.get()) == 0:
                savefilename = 'gen'
            else:
                savefilename = self.savefilename_entry.get()
            f.write(f'savefilename {savefilename}\n')
            f.write(f'show_NNC {self.show_nnc.get()}\n')
            f.write(f'mode {self.mode_combobox.get()}\n')

    def _calculate(self):
        calculate(self._get_config_filename())

    def _set_entry(self, entry: CTk.CTkEntry, text: str):
        if len(text) != 0:
            entry.delete(0, CTk.END)
            entry.insert(0, text)

    def _set_entry_from_content(self, entry: CTk.CTkEntry, content: list[str], i: int):
        if len(content) > 1:
            self._set_entry(entry, content[i])

    def _set_checkbox(self, checkbox: CTk.CTkCheckBox, value: str):
        if value == '1':
            checkbox.select()
        else:
            checkbox.deselect()

    def _get_config_filename(self):
        if len(self.config_filename.get()) == 0:
            filename = 'config.txt'
        else:
            filename = self.config_filename.get() + '.txt'

        return filename

    def _read_config(self):
        with open(self._get_config_filename(), 'r') as f:
            contents = f.readlines()
            for content in contents:
                content = content.split()
                if len(content) == 0:
                    continue
                match content[0]:
                    case 'model':
                        self.model_combobox.set(content[1])
                    case 'ndim':
                        self.dim_combobox.set(content[1])
                    case 'range':
                        if len(content) > 1:
                            self._set_entry(self.range0_entry_x, content[1])
                        if len(content) > 2:
                            self._set_entry(self.range0_entry_y, content[2])
                        if len(content) > 3:
                            self._set_entry(self.range0_entry_z, content[3])
                    case 'azimuth':
                        if len(content) > 1:
                            self._set_entry(self.azimuth_entry_x, content[1])
                        if len(content) > 2:
                            self._set_entry(self.azimuth_entry_y, content[2])
                        if len(content) > 3:
                            self._set_entry(self.azimuth_entry_z, content[3])
                    case 'c0':
                        self._set_entry_from_content(self.c0_entry, content, 1)
                    case 'alpha':
                        self._set_entry_from_content(self.alpha_entry, content, 1)
                    case 'nugget':
                        self._set_entry_from_content(self.nugget_entry, content, 1)
                    case 'wradius':
                        self._set_entry_from_content(self.wradius_entry, content, 1)
                    case 'lookup':
                        self._set_checkbox(self.lookup_checkbox, content[1])
                    case 'nb':
                        self._set_entry_from_content(self.nb_entry, content, 1)
                    case 'nx':
                        self._set_entry_from_content(self.grid_entry_x, content, 1)
                    case 'ny':
                        self._set_entry_from_content(self.grid_entry_y, content, 1)
                    case 'nz':
                        self._set_entry_from_content(self.grid_entry_z, content, 1)
                    case 'm':
                        self._set_entry_from_content(self.m_entry, content, 1)
                    case 'mean':
                        self._set_entry_from_content(self.mean_entry, content, 1)
                    case 'nnc':
                        self._set_checkbox(self.nnc_checkbox, content[1])
                    case 'category':
                        self._set_checkbox(self.cat_checkbox, content[1])
                    case 'cat_threshold':
                        to_view = ''
                        for i_th in range(1, len(content)):
                            to_view += ' ' + str(content[i_th])
                        self._set_entry(self.cat_threshold_entry, to_view)
                    case 'debug':
                        self._set_checkbox(self.debug_checkbox, content[1])
                    case 'calc_frob':
                        self._set_checkbox(self.frob_checkbox, content[1])
                    case 'seed_search':
                        self._set_entry_from_content(self.seed_search_entry, content, 1)
                    case 'seed_path':
                        self._set_entry_from_content(self.seed_path_entry, content, 1)
                    case 'seed_U':
                        self._set_entry_from_content(self.seed_U_entry, content, 1)
                    case 'cutoff':
                        self._set_entry_from_content(self.cutoff_entry, content, 1)
                    case 'bins':
                        self._set_entry_from_content(self.bins_entry, content, 1)
                    case 'show':
                        self._set_checkbox(self.screen, content[1])
                    case 'save':
                        self._set_checkbox(self.save, content[1])
                    case 'savefilename':
                        if content[1] != 'gen':
                            self._set_entry_from_content(self.savefilename_entry, content, 1)
                    case 'show_NNC':
                        self._set_checkbox(self.show_nnc, content[1])
                    case 'mode':
                        self.mode_combobox.set(content[1])



