import matplotlib.pyplot as plt
import os 

class Plotter:
    def __init__(self, exp_path) -> None:
        self.path = exp_path
    
    def _get_run_data(self, problem: str, param: int, filename:str)->dict:
        models = dict()
        problem_path = os.path.join(self.path, problem, param)
        used_models = os.listdir(problem_path)
        for model in used_models:
            model_path = os.path.join(problem_path, model)
            model_file = None
            for root, _, files in os.walk(model_path):
                for f in files:
                    if f == filename:
                        model_file = os.path.join(root, f)
                        break
            if model_file is None:
                print(f"{filename} not found under {model_path}")
                continue
            
            with open(model_file, "r") as source:
                lines = source.readlines()
            model_data = [float(i.split(",")[-1]) for i in lines]
            models[model] = model_data
        return models
    
    def plot_run(self, models_data:dict):
        for model in models_data.keys():
            plt.plot(models_data[model], label=model)
        plt.title = "Loss values over training"
        plt.xlabel = "Epoch"
        plt.ylabel = "Loss value"
        plt.legend()




if __name__ == '__main__':
    plotter = Plotter("storage")
    data = plotter._get_run_data("copy", 2000, "CE.txt")
    plotter.plot_run(data)