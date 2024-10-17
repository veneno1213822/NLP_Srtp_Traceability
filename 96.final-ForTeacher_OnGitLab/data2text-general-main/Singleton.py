# 训练和推理不能同时，训练时请求推理会直接return；训练时不能再训练，重叠训练也会直接return
class Singleton_IfTra:  # 单例模式，即确保一个类在整个程序生命周期中只存在一个实例
    _instance = None
    shared_var = False  # "initial value"
    def __new__(cls):  # 类的构造方法，在调用 __init__ 之前被调用，用于创建并返回这个类的实例。
        if cls._instance is None:
            cls._instance = super(Singleton_IfTra, cls).__new__(cls)
        return cls._instance