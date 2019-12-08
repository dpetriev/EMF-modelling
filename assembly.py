#
# GUI для расчета поля конденсатора
# 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.linalg import solve, cond
from datetime import datetime
import timeit
from sklearn.linear_model import Ridge, ridge_regression
from scipy.special import ellipeinc, ellipk

from ipywidgets import interactive
from IPython.display import display, clear_output
from ipywidgets import FloatSlider, Button, Output, Label
from ipywidgets import FloatRangeSlider, FloatSlider, IntSlider
from ipywidgets import HBox, VBox, Output, HTML 
from ipywidgets import ToggleButtons, RadioButtons, SelectionSlider
from ipywidgets import HTMLMath, Tab, Layout, ToggleButtons, Textarea
from time import sleep, time 
import datetime


from charges_class import Charges as chg
chg.dim = 2


class Tab0:
    '''
    Класс с графическим пользовательским интерфейсом для задания параметров
    '''
    def update_label(self, key, value, el):
        v = '' if value is None else '{0:6.3e}'.format(value)
        l = {'cond': 'Число обусловленности: ',
            'err': 'Погрешность: ',
            't': 'Время счета (c): ',
            'E': 'Emax/Emin: '}
        el.value = l[key] + v

    def send_message(self, mes):
        message = '<b>{}</b>'.format(mes)
        self.l_msg.value = message

    def runtime(self, start=True):
        if start:
            self.start_time  = time()
            return None
        else:
            self.finish_time = time()
            self.duration = self.finish_time - self.start_time
            return self.duration

    def get_data(self):
        dim = int(self.w_dim.value)
        chg.dim = dim
        r0 = float(self.w_r0.value)
        chg.r0 = r0
        alpha = float(self.w_alpha.value)
        chg.alpha = alpha
        n = int(self.w_n.value)
        chg.n = n

        minmaxx = self.w_minmaxx.value
        minmaxy = self.w_minmaxy.value
        minmax = (minmaxx[0], minmaxy[0], minmaxx[1], minmaxy[1])
        chg.minmax = minmax

        areax = self.w_minmaxxarea.value
        areay = self.w_minmaxyarea.value
        area = (areax[0], areay[0], areax[1], areay[1])
        chg.minmaxarea = area
        # with self.out0:
        #     print('chg.area', chg.area)

        na = self.w_na.value
        chg.na = na
        variant_dsc = self.w_dsc.value
        chg.variant_dsc = variant_dsc 

        return dim, r0, alpha, n, minmax, area, variant_dsc


    def __init__(self):

        # размерность задачи
        items_layout = Layout (flex = '1 1 auto', width = 'auto')

        self.w_title = HTML('<h3>Параметры задачи</h3>')
        self.w_l_dim = Label('Размерность задачи:')
        self.w_dim = ToggleButtons(
            options=['2', '3'],
            value = '2',
            #description=' ',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Плоская двухмерная задача', 'Трехмерная осесимметричная задача'],
            layout=items_layout
            )

        chg.r0=1e-6
        self.w_l_r0 = Label('Радиус заряда:')
        self.w_r0 =SelectionSlider(
            options=['1e-6', '1e-5', '1e-4', '1e-3', '1e-2'],
            value='1e-6',
            #description=' ',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True, layout=items_layout
        )   

        chg.alpha = 0
        self.w_l_alpha = Label(r"Параметр регуляризации ($\alpha$):")
        self.w_alpha =SelectionSlider(
            options=['0', '1e-6', '1e-4', '1e-2', '1e-1', '0.5', '1.0'],
            value='0',
            #description=' ',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True, layout=items_layout
        ) 

        chg.n = 100
        self.w_l_n = Label('Число разбиений области:')
        self.w_n = SelectionSlider(
            options=['20', '50', '100', '200', '400', '500', '1000'],
            value='100',
            #description=' ',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True, layout=items_layout
        )

        chg.minmaxxy=(0., 0, 1.,1.)
        self.w_l_minmaxxy = Label('Область для расчета и отображения потенциала (x, y):')
        self.w_minmaxx = FloatRangeSlider(value=[0.,1.],
            min=0,
            max=2.0,
            #description='xmin, xmax',
            step=0.01,    
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f', layout=items_layout)
        self.w_minmaxy = FloatRangeSlider(value=[0.,1.],
            min=0,
            max=2.0,
            #description='ymin, ymax',
            step=0.02,    
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f', layout=items_layout)

        chg.minmaxarea=(0.2, 0.2, 0.9,0.9)
        self.w_l_minmaxarea = Label('Область для расчета напряженности (x, y):')
        self.w_minmaxxarea = FloatRangeSlider(value=[0.2,0.9],
            min=0,
            max=1.0,
            #description='xmin, xmax',
            step=0.01,    
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f', layout=items_layout)
        self.w_minmaxyarea = FloatRangeSlider(value=[0.2,0.9],
            min=0,
            max=1.2,
            #description='ymin, ymax',
            step=0.005,    
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.3f', layout=items_layout)            
        
        self.w_l_na = Label('Разбиений для аппроксимации электродов:')
        chg.na = 50
        self.w_na = IntSlider(
            value=chg.na,
            min=20,
            max=250,
            step=2,
            #description=' ',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d', layout=items_layout
        )

        chg.c = None
        self.l_c = Label('')
        self.update_label('cond', chg.c, self.l_c)

        chg.err = None
        self.l_err = Label('')
        self.update_label('err', chg.err, self.l_err)

        chg.E = None
        self.l_E = Label('')
        self.update_label('E', chg.E, self.l_E)

        chg.t = None
        self.l_t = Label('')
        self.update_label('t', chg.t, self.l_t)

        self.l_msg=HTML('<b></b>')

        self.l_var = Label('Вариант:')
        self.w_dsc = Textarea(value='',
                placeholder='Введите пояснения по варианту расчета',
                description='',
                disabled=False
        )
        

        self.btn0 = Button(description ='Пересчитать поле', button_style='primary')
        self.btn01 = Button(description ='Emax/Emin', button_style='primary', disabled=True)
        self.out0 = Output()
        self.btn02 = Button(description='Сохранить', disabled=False)
        vb01= VBox([self.w_l_dim, self.w_dim,
                    self.w_l_r0, self.w_r0,
                    self.w_l_alpha, self.w_alpha,
                    self.w_l_n, self.w_n,
                    self.w_l_minmaxxy, self.w_minmaxx, self.w_minmaxy,
                    self.w_l_minmaxarea, self.w_minmaxxarea, self.w_minmaxyarea], 
                    layout=Layout(width='47%'))
        vb02 = VBox([self.w_l_na, self.w_na, self.l_var, self.w_dsc,
                     self.l_c, self.l_err, self.l_E , self.l_t, self.l_msg ], layout=Layout(width='47%'))
       
        hbox =HBox([vb01, vb02], 
                          layout=Layout(justify_content='space-between'))
        hbox2 = HBox([self.btn0, self.btn01, self.btn02])                  
        self.layout = VBox([self.w_title, hbox, hbox2, self.out0], layout=Layout(width='100%'))
        # инициализация
        self.get_data()


        

class Tab1:
    '''
    Верхняя изогнутая  часть электрода
    '''
    def __init__(self):
        items_layout = Layout (flex = '1 1 auto', width = 'auto')
        self.w_title = HTML('<h3>Форма электродов</h3>')
        self.w_l_x = Label('Границы изогнутой части электрода по горизонтали:')

        self.w_x = FloatRangeSlider(value=[0.5,1.], min=0., max=1., step=0.02,
                    continuous_update=False,
                    readout_format='.3f', layout=items_layout)
             
        self.w_l_y = Label('Координаты электрода по вертикали:')
        self.w_l_y0 = Label('y0')

        layouth = Layout(height='300px')
        self.w_y0 = FloatSlider(value=0.6,
            min=0.5,
            max=1.0,
            step=0.001,
            description='y0',
            disabled=False,
            continuous_update=False,
            orientation='vertical',
            readout=True,
            readout_format='.3f', layout=layouth
        )
        
        self.w_y1 = FloatSlider(value=0.6,
            min=0.5,
            max=1.0,
            step=0.001,
            description='y1',
            disabled=False,
            continuous_update=False,
            orientation='vertical',
            readout=True,
            readout_format='.3f', layout=layouth
        )
        
        self.w_y2 = FloatSlider(value=0.6,
            min=0.5,
            max=1.0,
            step=0.001,
            description='y2',
            disabled=False,
            continuous_update=False,
            orientation='vertical',
            readout=True,
            readout_format='.3f', layout=layouth
        )
        
        self.w_y3 = FloatSlider(value=0.6,
            min=0.5,
            max=1.0,
            step=0.001,
            description='y3',
            disabled=False,
            continuous_update=False,
            orientation='vertical',
            readout=True,
            readout_format='.3f', layout=layouth
        )
        
        self.w_y4 = FloatSlider(value=0.6,
            min=0.5,
            max=1.0,
            step=0.001,
            description='y4',
            disabled=False,
            continuous_update=False,
            orientation='vertical',
            readout=True,
            readout_format='.3f', layout=layouth
        ) 

        self.out_el = Output()

        hbox1 = HBox([self.w_y0, self.w_y1, self.w_y2, self.w_y3, self.w_y4], layout=Layout(width='100%'))
        vbox1 = VBox([self.w_l_x,  self.w_x, self.w_l_y, hbox1], layout=Layout(width='50%', padding='5px'))
        vbox2 = HBox([self.out_el], layout=Layout(width='50%', padding='5px'))
        hbox = HBox ([vbox1, vbox2], layout=Layout(width='100%'))

        self.layout = VBox([self.w_title, hbox])
        self.interactive_el_form = interactive(self.el_form, x0xm=self.w_x, y0=self.w_y0, y1=self.w_y1,
                                 y2=self.w_y2, y3=self.w_y3, y4=self.w_y4)


    def el_form( self, x0xm, y0, y1, y2, y3, y4):
        x0, xm = x0xm
        dy = y0 - 0.5
        ymin= 0.5
        ymax = y0 + dy
        w_y = [self.w_y0, self.w_y1, self.w_y2, self.w_y3, self.w_y4]
        
        # for y in range(w_y):
        #     y.min = ymin
        #     y.max = ymax
        #     with self.out_el:
        #         print(y0, dy, ymin, ymax)


        x = np.linspace(x0, xm, 5)
        y = np.array([y0, y1, y2, y3, y4])

        with self.out_el:            
            clear_output(True)
            #print('w_y', w_y)
            for w in w_y:
                dy = y0 - 0.5
                w.max = y0+dy
                #print('w', w.min, w.max)

            # аппроксимация электродов
            o  = chg.fit_electrodes(chg.na, x, y)
            chg.qxy, chg.upperx, chg.uppery, chg.lowerx, chg.lowery = o
            chg.fig1=chg.electrodes_show(chg.na, x, y, scale=True) 
            #print('electrodes na qxy', chg.na, len(chg.qxy) )
            plt.show()

    def get_elecrode_data(self):
        return self.w_x.value, [self.w_y0.value, self.w_y1.value, 
               self.w_y2.value, self.w_y3.value, self.w_y4.value]

class Tab2:
    '''
    Вкладка с распределением зарядов
    '''
    def __init__(self):
        self.w_title = HTML('<h3>Распределение зарядов на электродах</h3>')
        self.w_out2 = Output()  
        self.layout = VBox([self.w_title, self.w_out2])


    def charges_on_electrodes(self):
        # распределение зарядов  на электродах
        chg.qxy, chg.c, chg.err = chg.calc_charges(chg.qxy, alpha=chg.alpha, r0=chg.r0)
        with self.w_out2:
            #print('qxy=', chg.qxy)
            #print('*dim=', chg.dim, 'cond=', chg.c, 'err=', chg.err)
            clear_output(True)
            chg.fig2 = chg.draw_charges_from_qxy(chg.qxy)
            plt.show()


class Tab3:
    '''
    Распределение поля  в конденсаторе
    '''
    def __init__(self):
        self.w_title = HTML('<h3>Распределение поля в конденсаторе</h3>')
        self.w_out3 = Output()  
        self.layout = VBox([self.w_title, self.w_out3])

    def calc_field(self):
        with self.w_out3:
            chg.EmaxEmin, pp, E,  chg.fig3 = chg.calculate(chg.qxy, minmax=chg.minmax, n=chg.n, 
                                                    r=chg.r0, cmap=None, alpha=0.5, 
                                                    figsize=(8,6), levels=20, 
                                                    axis=True, area=chg.minmaxarea, fn='', ms=3)
            chg.EmaxEmin, pp, E,  chg.figx = chg.calculate(chg.qxy, minmax=chg.minmaxarea, n=chg.n, 
                                                    r=chg.r0, cmap=None, alpha=0.5, 
                                                    figsize=(8,6), levels=20, draw = False,
                                                    axis=True, area=None, fn='', ms=3)

            #print('chg.EmaxEmin nm fig3', chg.EmaxEmin, chg.minmaxarea, len(chg.qxy), chg.fig3)
        return chg.fig3

    def draw_field(self):
        with self.w_out3:
            clear_output(True)
            #print('area', chg.minmaxarea)
            chg.fig3 = self.calc_field()
            plt.show()



class Tab4:
    '''
    Распределение поля  в выделенной области
    '''
    def draw_field(self):
        with self.w_out4:
            #print('area before:', chg.minmaxarea)
            clear_output()
            chg.EmaxEmin, pp, E,  chg.fig4 = chg.calculate(chg.qxy, minmax=chg.minmaxarea, n=chg.n, 
                                                    r=chg.r0, cmap=None, alpha=0.5, 
                                                    figsize=(8,6), levels=20, draw = True,
                                                    axis=True, area=None, fn='', ms=3)

            #print('chg.EmaxEmin nm', chg.EmaxEmin, chg.minmaxarea, len(chg.qxy), chg.fig4)
        return chg.fig4


    def __init__(self):
        self.w_title = HTML('<h3>Распределение поля в выделенной области</h3>')
        self.w_out4 = Output()  
        self.layout = VBox([self.w_title, self.w_out4])

class Tab5:
    '''
    Неравномерность поля в выделенной области
    '''
    def __init__(self):
        self.dx, self.dy = 0.005, 0.005
        self.w_title = HTML("<h3>Область с заданной неравномерностью </h3>")
        self.w_out5 = Output()  
        self.w_label_var = Label("Допустимая неравномерность:")
        self.w_var = FloatSlider(1.1, min=1., max=1.5, step=0.005, continuous_update=False,  readout_format='.3f')
        self.w_label_xarea = Label('Границы допустимой области по горизонтали: ')
        self.w_xarea = FloatRangeSlider(value=[0.2,0.7],
            min=0,
            max=1.0,
            step=0.005,    
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.3f')
        self.w_label_yarea = Label('Границы допустимой области по вертикали: ')
        self.w_yarea = FloatRangeSlider(value=[0.45,0.55],
            min=0,
            max=1.0,
            step=0.005,    
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.3f')
        self.w_varn = Label()
        self.btn_var = Button(description ='Пересчитать', button_style='primary', disabled = True) # доработать
        self.btn_break = Button(description ='Прервать',  disabled = True)
        self.w_message = HTML("")
        self.w_result = HTML("")
        self.layout = VBox([self.w_title, self.w_label_var, self.w_var, 
        self.w_label_xarea,  self.w_xarea,
        self.w_label_yarea,  self.w_yarea,
        self.w_varn,
        self.btn_var,  self.w_message, self.w_result,
        self.w_out5])
        self.btn_var.on_click(self.button_var_onclick)
        
    def calc(self, var_ok,  var, minmaxarea, dx):
        '''
        Расчет области
        '''
        ok = True
        minmaxarea_n = (minmaxarea[0], minmaxarea[1], minmaxarea[2]+dx, minmaxarea[3]) 
        var_new, *rest = chg.calculate(chg.qxy, minmax=minmaxarea_n, n=chg.n, 
                                        r=chg.r0, draw = False,
                                        axis=True, area=None, fn='', ms=3)
        ok = var_new <= var_ok
        #with self.w_out5:
        #    print('!calc ok=',ok, 'dx=', dx, 'minmaxarea_n=', minmaxarea_n, 'var_new=', var_new)
        return var_new, minmaxarea_n, ok
 

    
    def  button_var_onclick(self, b):
        self.br = False
        self.w_result.value=''
        self.w_varn.value = ''
        self.w_message.value="<b>Считаю!</b>"
        self. count = 100
        var = 1e30
        direction = 'x'
        minmaxarea = (self.w_xarea.value[0], self.w_yarea.value[0], self.w_xarea.value[1], self.w_yarea.value[1])
        var, *rest = chg.calculate(chg.qxy, minmax=minmaxarea, n=chg.n, 
                            r=chg.r0, draw = False,
                            axis=True, area=None, fn='', ms=3)
        ok_var = self.w_var.value

        if var<=ok_var:
            for i in range(self.count):
                var_n, minmaxarea, ok = self.calc(ok_var, var,  minmaxarea, self.dx)
                if ok:
                    var = var_n
                    self.w_xarea.value = [minmaxarea[0], minmaxarea[2]] 
                    self.w_yarea.value = [minmaxarea[1], minmaxarea[3]] 
                    self.w_varn.value='Текущая неравномерность:{0:6.3e}'.format(var)
  
                    self.w_message.value='<b>Итерация: {0}, неравномерность:{1:6.3e}</b>'.format(i, var)
                else:
                    #with self.w_out5:
                    #    print(' break i=', i, 'area=', minmaxarea)
                    break
        if var>ok_var:
            for i in range(self.count):
                var_n, minmaxarea, ok = self.calc(ok_var, var,  minmaxarea, -self.dx)
                var = var_n
                self.w_xarea.value = [minmaxarea[0], minmaxarea[2]] 
                self.w_yarea.value = [minmaxarea[1], minmaxarea[3]] 
                self.w_varn.value='Текущая неравномерность:{0:6.3e}'.format(var)
  
                self.w_message.value='<b>Итерация: {0}, неравномерность:{1:6.3e}</b>'.format(i, var)
                #with self.w_out5:
                #    print('i=', i, 'area=', minmaxarea)
                if  ok:  
                    break


        if var<=ok_var:
            self.w_message.value='<b>Вычисление области с заданной неравномерностью успещно завершено!</b>'
            self.w_result.value='<b>Неравномерность:{0:6.3e} обеспечена в прямоугольнике:({1:5.3f}, {2:5.3f}, {3:5.3f}, {4:5.3f})</b>'.\
                                format(var, minmaxarea[0], minmaxarea[1], minmaxarea[2], minmaxarea[3])
        else:
           self.w_message.value='<b>Подобрать область за заданное число итераций не удалось!</b>'


class Tabs:
    '''
    Собираю на вкладках пользователький интерфейс
    '''
    tab_names = ['Параметры', 'Электроды', 'Заряды', "Поле", "Поле в области", "Неравномерность"]
    tab_content = [VBox(), VBox(), VBox(), VBox(), VBox(), VBox()]

    def __init__(self):
        chg.dim = 2 # инициализация
        self.tabs = Tab()
        self.title = HTML('<h1>Выравнивание электрического поля в конденсаторе</h1>')
        self.label1 = Label('Допустимая неравномерность')

        children = [VBox(), VBox(), VBox(), VBox(), VBox(), VBox()]

        # Нулевая вкладка - параметры
        self.tab0 = Tab0()
        self.tab1 = Tab1()
        self.tab2 = Tab2()
        self.tab3 = Tab3()
        self.tab4 = Tab4()
        self.tab5 = Tab5()
        children[0] = self.tab0.layout
        children[1] = self.tab1.layout
        children[2] = self.tab2.layout
        children[3] = self.tab3.layout
        children[4] = self.tab4.layout
        children[5] = self.tab5.layout
        
        layout = self.tab0.layout.layout
        #layout.justify_content='space-between' # разношу столбцы

        # заношу вкладки
        self.tabs.children = children
        for i in range(len(Tabs.tab_names)):
            self.tabs.set_title(i, Tabs.tab_names[i])
        self.layout = VBox([self.title, self.tabs])

        # инициализация
        chg.c = None
        chg.err = None
        chg.EmaxEmin = None
        chg.t = None
        chg.n3 = 20 
        chg.dim = 2

        # обработчики
        self.tab0.btn0.on_click(self.btn0_click)
        self.tab0.btn01.on_click(self.btn01_click)
        self.tab0.btn02.on_click(self.btn02_click)

    # обработчики
    def btn0_click(self, b):
        with self.tab0.out0:            
            self.tab0.send_message('Подождите, считаю!')
            self.tab0.runtime(True) # запускаю счет
            # получаю данные из пользовательского интерфейса
            dim, r0, alpha, n, minmax, area, variant_dsc = self.tab0.get_data()
            #print('dim, r0, alpha, n, minmax, area, variant_dsc',
            #       dim, r0, alpha, n, minmax, area, variant_dsc)
            #print('btn0 clicked msg ', self.tab0.l_msg.value)

            # получаю данные о форме электродов
            #print('chg.dim, chg.na', chg.dim, chg.n)
            x0xm, yy = self.tab1.get_elecrode_data()
            #print('x0xm, yy', x0xm, yy)
            self.tab1.el_form(x0xm, *yy) # рисую электроды

            # заряды на электродах
            self.tab2.charges_on_electrodes()
            self.tab0.update_label('cond', chg.c, self.tab0.l_c)
            self.tab0.update_label('err', chg.err, self.tab0.l_err)

            # рисуем картину поля
            self.tab3.draw_field()
            self.tab0.update_label('E', chg.EmaxEmin, self.tab0.l_E)

            self.tab0.send_message('Вычисления завершены!')
            d = self.tab0.runtime(False)
            self.tab0.update_label('t', d, self.tab0.l_t)
            chg.t = d
            self.tab0.btn01.disabled=False
            self.tab0.btn02.disabled=False

            self.tab5.btn_var.disabled = False
            self.tab5.btn_break.disabled = False

        

    def btn01_click(self, b):
        with self.tab0.out0:
            self.tab0.runtime(True) # запускаю счет
            self.tab0.send_message('Подождите, считаю!')
            self.tab0.runtime(True) # запускаю счет
            dim, r0, alpha, n, minmax, area, variant_dsc = self.tab0.get_data()
            chg.minmaxarea = area
            # рисуем картину поля
            self.tab4.draw_field()
            self.tab0.update_label('E', chg.EmaxEmin, self.tab0.l_E)

            self.tab0.send_message('Вычисления завершены!')
            d = self.tab0.runtime(False)
            self.tab0.update_label('t', d, self.tab0.l_t)
            chg.t = d
        
    def btn02_click(self, b):
        with self.tab0.out0:
            self.tab0.runtime(True) # запускаю счет
            self.tab0.send_message('Подождите, сохряняю данные и рисунки!')
            #print('Печать')
            now = datetime.datetime.now()
            self.variant = '{0:%Y_%m_%d_%H_%M_%S_%f}'.format(now)
            fn = 'log/{}.txt'.format(self.variant)
            with open(fn, "w") as tf:  
                tf.write('\n')                                  
                tf.write("Вариант:{}\n".format(self.variant))

                tf.write(self.tab0.w_dsc.value)
                tf.write('\n\n')
                tf.write('Размерность задачи:{}\n'.format(chg.dim))
                tf.write('Радиус заряда:{}\n'.format(chg.r0))
                tf.write('Число разбиений области:{}\n'.format(chg.n))
                tf.write('Разбиений для аппроксимации электродов:{}\n'.format(chg.na))
                tf.write('Число зарядов на обкладках конденсатора:{}\n'.format(len(chg.qxy)))
                tf.write('Параметр регуляризации:{}\n'.format(chg.alpha))
                tf.write("Область отображения поля:\n({},{},{}, {})\n".format(chg.minmax[0], chg.minmax[1],
                         chg.minmax[2], chg.minmax[3]))
                tf.write("Выделенная область для расчета напряженности поля:\n({},{},{}, {})\n".format(chg.minmaxarea[0], 
                    chg.minmaxarea[1], chg.minmaxarea[2], chg.minmaxarea[3]))
                tf.write('Число обусловленности:{0:6.3e}\n'.format(chg.c))
                tf.write('Погрешность решения сиcтемы уравнений:{0:6.3e}\n'.format(chg.err))
                tf.write('Время счета:{0:6.3e}\n'.format(chg.t))
                # сохраняю рисунки
                if chg.fig1:
                    chg.fig1.savefig('log/{}_fig1.png'.format(self.variant), dpi=600)

                if chg.fig2:
                    chg.fig2.savefig('log/{}_fig2.png'.format(self.variant), dpi=600)

                if chg.fig3:
                    chg.fig3.savefig('log/{}_fig3.png'.format(self.variant), dpi=600)

                if chg.fig4:
                    chg.fig4.savefig('log/{}_fig4.png'.format(self.variant), dpi=600)
                
    

            self.tab0.variant = self.variant
            self.tab0.l_var.value = 'Вариант: {0}'.format(self.variant) 
            self.tab0.send_message('Подождите, сохраняю данные!')
            #sleep(2)

            self.tab0.send_message('Данные сохранены')
            d = self.tab0.runtime(False)
            self.tab0.update_label('t', d, self.tab0.l_t)
            chg.t = d
            #print('Печать ')
   

if __name__=='__main__':
    tabs = Tabs()
    
    print('Finished')
