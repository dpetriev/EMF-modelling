import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.linalg import solve, cond
from datetime import datetime
import timeit
from sklearn.linear_model import Ridge, ridge_regression
from scipy.special import ellipeinc, ellipk


class Charges:
    '''
    Поле системы зарядов
    '''
    # dim =2 для двухмерной задачи и dim=3 для трехмерной осесимметричной задачи
    dim = 0 # размерность задачи

    @staticmethod
    def set_dim(dim):
        # задать размерность
        Charges.dim = dim
        Charges.check_dim()

    @staticmethod
    def check_dim():
        if Charges.dim not in (2,3):
            raise ValueError('Некорректная размерность задачи dim={}'.format(Charges.dim))

    @staticmethod
    def coverage(n, xmin=0., ymin=0., xmax=1., ymax=1.):
        '''
        Покрытие прямоугольника (xmin, ymin, xmax, ymax) сеткой
        с n разбиениями
        Метод возвращает  массив  с координатами точек покрытия
        '''
        x = np.linspace(xmin,xmax, n+1)
        y = np.linspace(ymin,ymax, n+1)
        X, Y = np.meshgrid(x, y)
        xx = X.reshape(-1,1)
        yy = Y.reshape(-1,1)
        return np.hstack([xx, yy])

    @staticmethod
    def dist(i, xy, qxy, r=0.02):
        '''
        Расстояние от i-ого заряда до покрытия xy
        r - радиус заряда
        Метод возвращает массив расстояний
        '''
        d = xy - qxy[i, 1:]
        dst = np.sqrt(np.sum(d**2,axis=1))
        dst = np.where(np.abs(dst)<=r, r, dst)
        return dst


    @staticmethod
    def potential(qxy, xy, r=0.01, eps=1e-6):
        '''
        Потенциал от системы зарядов qxy в точках покрытия xy
        r - радиус заряда.
        Метод возвращает  одномерный массив потенциалов на покрытии xy
        '''
        Charges.check_dim()
        n, m = xy.shape
        mq = len(qxy)
        p = np.zeros(n)
        if Charges.dim==2:
            for i in range(mq):
                c = Charges.dist(i, xy, qxy, r=r)
                p -= qxy[i, 0]*np.log(c)
            return p # 2*p
        if Charges.dim==3:
            # x здесь r, r,  y переименовываем в z
            r, zr = xy[:,0], xy[:,1]
            R0 = eps
            #r = np.where(r>eps, r, eps)
            for i in range(mq):
                # цикл по зарядам
                qq, R, zR= qxy[i,0], qxy[i, 1], qxy[i,2]
                R = np.where(R>eps, R, eps)
                z = np.abs(zr-zR)
                s = np.sqrt((R+r)**2+z**2)
                phi = np.where(np.logical_and(np.logical_and(z<=R0, r<=R0), R<=R0), 0., R/s*ellipk(2.*np.sqrt(r*R)/s)) 
                p+=qq*phi
            return p 

    @staticmethod
    def potentialy(qxy, xx, yy, r = 1e-2, m = 200):
        '''
        Потенциал от системы зарядов  qxy,
        в сечении в точке по оси абсцисс xx,
        yy = (ymin, ymax),
        r - радиус электрода,
        m - число разбиений по оси y
        функция возвращает данные для рисования:
        координаты по оси ординат,
        значения потенциала

        '''
        y = np.linspace(yy[0], yy[1], m+1)
        xy =np.zeros((m+1, 2))
        xy[:,0] = xx
        xy[:,1] = y
        p = Charges.potential(qxy, xy, r=r)
        return y, p

    @staticmethod
    def potentialx(qxy, xx, yy, r = 1e-2, m = 200):
        '''
        Потенциал от системы зарядов  qxy,
        в сечении в точке по оси ординат yy,
        xx = (xmin, xmax),
        r - радиус электрода,
        m - число разбиений по оси y
        функция возвращает данные для рисования:
        координаты по оси абсцисс,
        значения потенциала

        '''
        x = np.linspace(xx[0], xx[1], m+1)
        xy =np.zeros((m+1, 2))
        xy[:,0] = x
        xy[:,1] = yy
        p = Charges.potential(qxy, xy, r=r)
        return x, p

    @staticmethod
    def p_show(x, y, p, figsize=(8,6), levels=20, cmap='binary', alpha=0.8, 
               density=0.5, fn='', fig=None, axis=None):
        '''
        Отображаем поле зарядов
        x, y - разбиения по осям координат
        p - массив потенциалов
        figsize - размер рисунка
        levels - число уровней
        cmap - палитра
        alpha - непрозрачность
        density - плотность линий тока
        fn - имя файла для сохранения картинки
        Метод возвращает объект рисунка
        '''
        #fig = plt.figure(figsize=figsize)
        m, n  = x.shape[0], y.shape[0] 
        pp = p.reshape((m, n))
        X, Y = np.meshgrid(x, y)
        if cmap:
            cb = plt.contourf(x, y, pp, cmap=cmap, alpha=alpha)  
            plt.colorbar(cb)  
        gy, gx = np.gradient(pp)
        plt.streamplot(X, Y, gx, gy, color='black', density=density)        
        c = plt.contour(X, Y, pp, levels, colors='black', linestyles='solid')        
        cl = plt.clabel(c, fmt='%1.2f')        
        
        if not axis:
            plt.axis('off')
        
        if fn:
            plt.savefig(fn+'.png', dpi=600)
        #return fig

    @staticmethod
    def appr(x, a3, a4, a5):
        '''
        Вычисление аппроксимации верхней правой четвертушки электрода
        полиномом  червертой степени
        x - координата по оси абсцисс
        a0, ... a4 - коэффициенты полинома
        Функция возвращает  координату электрода по оси y
        '''
        # global x0, y0 
        x0, y0 = Charges.x0, Charges.y0
        xx = x - x0
        return np.where(x<=x0, y0, y0 + a3*xx**3 + a4*xx**4 + a5*xx**5)


    @staticmethod
    def fit_electrodes(n, x, y):
        '''
        По координатам четвертушки (верхней правой) электрода x, y
        аппроксимирует конденсатор:
        1) для трехмерной осесимметричной задачи справа от вертикальной оси (2*n) 
        2) для двухмерной плоской задачи весь конденсатор (4*n)
        Функция возвращает:
        qxy - заряды и координаты  электродов в формате [(q0, x0, y0), (q1, x1, y1), ...(q2n,x2n, y2n)]
        upperx, uppery, lowerx, lowery координаты по x, y верхнего и нижнего электродов    
        '''
        # начало изогнутой части электрода
        x0, y0 = x[0], y[0]
        Charges.x0, Charges.y0 = x0, y0 # сохраняю в атрибутах класса
        # аппроксимация
        g, h = curve_fit(Charges.appr, x, y)
        xa = np.linspace(0, 1, n)
        ya = Charges.appr(xa, *g)
        if Charges.dim==3:
            upperx = xa
            uppery = ya
            lowerx = xa
            lowery = 1.- ya
            qxy =[]
            for i in range(len(upperx)):
                qxy.append((1., upperx[i], uppery[i]))
            for i in range(len(lowerx)):
                qxy.append((-1., lowerx[i], lowery[i]))
            qxy =np.array(qxy)
            return qxy, upperx, uppery, lowerx, lowery
        if Charges.dim==2:
            upperx = np.hstack((xa[:-1]/2., 0.5+xa/2.))
            uppery = np.hstack((ya[-1:0:-1], ya))
            lowerx = upperx
            lowery = np.hstack((1-ya[-1:0:-1], 1-ya))
            qxy = []
            for i in range(len(upperx)):
                qxy.append((1., upperx[i], uppery[i]))
            for i in range(len(lowerx)):
                qxy.append((-1., lowerx[i], lowery[i]))
            qxy = np.array(qxy)
            return qxy, upperx, uppery, lowerx, lowery

    @staticmethod
    def electrodes_show(n, x, y, figsize=(8,6), fn=None, scale=True, dpi=96):
        '''
        Рисование электродов после аппроксимации верхней правой четвертушки
        n - разбиений четвертушки
        x, y - координаты четвертушки
        figsize - размеры графика
        fn - имя файла для сохранения графика
        '''
        qxy, upperx, uppery, lowerx, lowery = Charges.fit_electrodes(n, x,y)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(upperx, uppery, 'k-', lw=3)
        plt.plot(lowerx, lowery, 'k-', lw=3)
        plt.plot(x, y, 'ro', ms=5)
        if not scale:
            plt.ylim(0,1)
            plt.xlim(0,1)
        plt.xlabel('x', fontsize=14)
        plt.ylabel('y', fontsize=14)
        if fn:
            plt.savefig(fn+'.png', dpi=600)
        return fig        

    @staticmethod
    def calculate(qxy, minmax=(0,0, 1,1), n = 100, r = 1e-6, figsize=(7,6), levels=20, cmap='binary',
               alpha=0.8, density=0.5, fn='', draw=True, area=None, axis=None, ms=5):
        '''
        По зарядам и расположению электродов рассчитываем:
        поле и градиенты в вделенной области
        qxy - заряды и координаты электродов
        minmax -  границы отображаемой области 
        n - число разбиений по осям
        r - радиус заряда (параметр регуляризации)
        figsize - размер рисунка
        levels - число уровней
        cmap - палитра
        alpha - непрозрачность
        density - частота линий тока
        fn - имя файла для сохранения рисунка
        draw - флажок рисования
        area - границы выделенной области
        axis - отображение оцифровки
        '''

        x = np.linspace(minmax[0], minmax[2], n+1)
        y = np.linspace(minmax[1], minmax[3], n+1)
        xy = Charges.coverage(n, xmin=minmax[0], ymin=minmax[1], xmax=minmax[2], ymax=minmax[3])
        p = Charges.potential(qxy, xy,r=r)
        fig = None
        if draw:
            fig = plt.figure(figsize=figsize)
            Charges.p_show(x, y, p, figsize=figsize, levels=levels, cmap=cmap, 
                        alpha=alpha, density=density, fn=fn, axis=axis)
            # отображение электродов
            xmin, xmax = plt.xlim()[0], plt.xlim()[1]
            ymin, ymax = plt.ylim()[0], plt.ylim()[1]

            plt.plot(qxy[:,1], qxy[:,2], 'bo', ms=ms)
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            if area:
                xmin, ymin, xmax, ymax = area
                plt.plot([xmin, xmin], [ymin, ymax], 'r-')
                plt.plot([xmax, xmax], [ymin, ymax], 'r-')
                plt.plot([xmin, xmax], [ymin, ymin], 'r-')
                plt.plot([xmin, xmax], [ymax, ymax], 'r-')
            plt.show()

        # разбираемся с градиентами и напряженностью
        nxy = int(np.sqrt(p.shape[0]))
        pp = p.reshape(nxy,nxy)
        gy, gx = np.gradient(pp, x, y)
        E = np.sqrt(gy**2 + gx**2)
        Emin, Emax = np.min(E), np.max(E)
        EmaxEmin = Emax/Emin if abs(Emin)>1.e-10 else np.inf
        # print('E', EmaxEmin, Emin, Emax)
        return EmaxEmin, pp, E,  fig


    @staticmethod
    def calc_charges(qxy, r0=0.000001, R0=0, alpha=0.):
        '''
        Вычисление  распределения зарядов (подход третий)
        qxy - исходное распределение зарядов и их кординаты 
        r0 - радиус заряда
        R0- расстояние от заряда до линии вычисления потенгциала
        alpha - параметр регуляризации
        функция возвращает перерассчитанный qxy и число обусловленности матрицы
        '''
        R0 = r0 if not r0<R0 else R0
        qq, xx, yy = qxy[:,0], qxy[:,1], qxy[:,2]
        q, x, y = qq.copy(), xx.copy(), yy.copy()
        n = len(q)
        nn = int(n/2)        
        # расставляю заряды        
        q[:nn] = 1. # верхний электрод
        q[nn:] = -1. # нижний электрод
        # выставляю линию  единичного потенциала - верхний электрод
        y[:nn] -= R0
        # высставляю линию нулевого потенциала
        y[nn:] += R0

        A = np.zeros((n, n))
        xy = np.hstack([x.reshape(n,1), y.reshape(n,1)])
        for i in range(n):
            A[i,:] = Charges.potential(qxy[i:i+1, :], xy, r=r0)

        # другой подход
        for i in range(n):
            for j in range(n):
                x1, y1, x2, y2 = qxy[i,1], qxy[i,2], qxy[j,1], qxy[j,2]
                if Charges.dim==2:
                    D = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    D = D if D>r0 else r0
                    A[i,j] = -np.log(D)
                if Charges.dim==3:
                    r, R = x1, x2
                    r = r if r>R0 else R0
                    R = R if R>R0 else R0
                    z = np.abs(y1-y2)
                    z = z if z>R0 else R0
                    s = np.sqrt((R+r)**2+z**2)
                    A[i,j] = 0. if z<=R0 and r<=R0 and R<=R0 else R/s*ellipk(2.*np.sqrt(r*R)/s) 
                    #print('i,j, A[i,j]', i,j, A[i,j] )

                    # nom = 2*r*R
                    # denom = r**2+R**2+z**2
                    # if denom <1e-8:
                    #     A[i,j] = 1.
                    # else:
                    #     A[i,j] = ellipeinc(np.pi/2, np.sqrt(2*r*R/(r**2+R**2+z**2)))
        #print('dim=',Charges.dim,'A=', A)
        # матрица расстояний

        # X1, X2 = np.meshgrid(x, x)
        # Y1, Y2 = np.meshgrid(y, y)
        # XX1, XX2 = np.meshgrid(xx, xx)
        # YY1, YY2 = np.meshgrid(yy, yy)

        # # попарные расстояние между зарядами
        # D = (X1-XX2)**2 +(Y1 - YY2)**2
        # # У нас не может быть нулевых расстояний, т.к.
        # # при расчете потенциалов приходится брать логарифм в двухмерном случае
        # D = np.where(D<r, r, D)
        # A = -np.log(D)
        c = cond(A) # число обусловленности
        # A*q = Fi - здесь  Fi - потенциалы на обкладках, например, 1 и -1
        # У нас как раз 1 и -1  для зарядов
        qxy[:,0] = ridge_regression(A, q, alpha)
        # проверка решения 
        u = A @  qxy[:,0]
        err = np.max(np.abs(u - q))
        #print('alpha =', alpha, 'err =', err, 'cond =', c, 'r =', r0, ' R=', R0)
        return qxy, c, err    


    @staticmethod
    def draw_elecrodes_from_qxy(qxy, figsize=(8,6), fn=None, ylim=None):
        '''
        Рисуем электроды,
        заданные  qxy
        '''
        fig = plt.figure(figsize=figsize)
        n = qxy.shape[0]
        n2 = int(n/2)
        plt.plot(qxy[:n2,1], qxy[:n2,2], 'k-', lw=3)
        plt.plot(qxy[n2:,1], qxy[n2:,2], 'k-', lw=3)
        plt.xlabel('x', fontsize=14)
        plt.ylabel('y', fontsize=14)
        if ylim:
            plt.ylim(ylim)
        if fn:
            plt.savefig(fn+'.png', dpi=600) 
        plt.show()
        return fig

    @staticmethod
    def draw_charges_from_qxy(qxy, figsize=(8,6), fn=None):
        '''
        Рисуем распределение зарядов на электродах
        '''
        ls = ('-','-.','--',':')
        fig = plt.figure(figsize=figsize)
        n = qxy.shape[0]
        n2 = int(n/2)
        plt.plot(qxy[:n2,1], qxy[:n2,0], 'k'+ls[0], lw=3, label='Верхний электрод')
        plt.plot(qxy[n2:,1], qxy[n2:,0], 'k'+ls[3], lw=3, label='Нижний электрод')
        plt.xlabel('x', fontsize=14)
        plt.ylabel('Распределение зарядов', fontsize=14)
        plt.legend(loc=0)
        if fn:
            plt.savefig(fn+'.png', dpi=600) 
        plt.show()
        return fig


if __name__=='__main__':
    print('Charges started')
    n, r, R, alpha  = 51, 1e-6, 1e-2, 0.
    # электроды
    x, y = [0.7, 0.8, 0.9, 1.0], [0.6, 0.6, 0.6, 0.6]
    # выделенная область
    xmin, ymin, xmax, ymax = 0., 0., 1., 1.
    minmax0=(0.,0.3, 1.2, 0.7)
    minmax1=(0.3,0.45, 0.9, 0.55)


    # размерность задачи
    Charges.set_dim(3)
    Charges.check_dim()

    # аппроксимация электродов
    qxy, upperx, uppery, lowerx, lowery = Charges.fit_electrodes(n, x, y)
    Charges.electrodes_show(n, x, y)
    plt.show()

    # распределение зарядов  на электродах
    qxy, c, err = Charges.calc_charges(qxy, alpha=alpha, r0=r)
    #print('qxy=', qxy)
    print('dim=', Charges.dim, 'cond=', c, 'err=', err)
    Charges.draw_elecrodes_from_qxy(qxy)
    Charges.draw_charges_from_qxy(qxy)

    # поле 
    EmaxEmin, p, *rest = Charges.calculate(qxy, minmax=minmax0, n=n, r=r, cmap=None, alpha=0.5, 
                      figsize=(8,6), levels=20, axis=True, area=minmax1, fn='dim2.png')
    print('EmaxEmin=', EmaxEmin)

    plt.show()

    print('Finished')

