from clearml import PipelineDecorator
from clearml import Task, Logger, Dataset

parameters = {
   'n_train' : 150,
   'n_test' : 1000,
   'noise' : 0.1,
   'max_depth': 5,
   'random_state': 17
}


@PipelineDecorator.component(cache=True, execution_queue="default")
def lib():
   import os
   import pandas as pd
   from pathlib import Path
   from sklearn.metrics import log_loss


@PipelineDecorator.component(cache=True, return_values=['path, labels, subs'], execution_queue="default")
def step_one(path):
   import os
   import pandas as pd

   submission = pd.read_csv(path / 'sample_submission.csv', index_col='id')
   labels = pd.read_csv(path / 'train_labels.csv', index_col='id')

   sub_ids = submission.index # the ids of the submission rows (useful later) [20000:40000]
   #gt_ids = labels.index # the ids of the labeled rows (useful later) [0:20000]

   #Добавим все файлы кроме 'sample_submission.csv', 'train_labels.csv'
   subs = [file for file in os.listdir(path) if file not in ['sample_submission.csv', 'train_labels.csv']]
   subs = sorted(subs)

   return labels, subs, sub_ids


@PipelineDecorator.component(return_values=['ensemble_data'], execution_queue="default")
def step_two(path, subs):
   import pandas as pd

   # Конкатинирование значений предсказаний из файлов в единый датасет для передачи в train_split
   ensemble_data = pd.concat([pd.read_csv(path / subs[i], index_col='id') for i in range(10)], axis=1)
   
   # Переименование дублирующихся столбцов
   ensemble_data.columns = [f"pred_{i}" for i in range(ensemble_data.shape[1])]

   return ensemble_data


@PipelineDecorator.component(return_values=['X_train, X_val, y_train, y_val'], execution_queue="default")
def step_three(ensemble_data, labels):
   from sklearn.model_selection import train_test_split

   X_train, X_val, y_train, y_val = train_test_split(ensemble_data[:20000], labels, test_size=0.2, random_state=42)

   return X_train, X_val, y_train, y_val


@PipelineDecorator.component(return_values=['y_pred_proba_val'], execution_queue="default")
def step_four(X_train, X_val, y_train, ensemble_data):
   from xgboost import XGBClassifier

   # Создание и обучение модели
   ensemble_model = XGBClassifier()
   ensemble_model.fit(X_train, y_train)

   # Предсказание на валидационном наборе
   y_pred_proba_val = ensemble_model.predict_proba(X_val)[:, 1]

   # Предсказание на тестовом наборе
   # Создание DataFrame из массива blade[20000:]
   y_pred_proba_test = ensemble_model.predict_proba(ensemble_data[20000:])[:, 1]

   return y_pred_proba_val, y_pred_proba_test


@PipelineDecorator.component(execution_queue="default")
def step_five(y_val, y_pred_proba_val):
   from sklearn.metrics import log_loss

   # Расчет log_loss
   source = log_loss(y_val, y_pred_proba_val)
   print("Source:", source)


@PipelineDecorator.component(execution_queue="default")
def step_six(y_pred_proba_test, sub_ids):
   import pandas as pd

   # Установка индексов, начиная с 20000
   #index_values = range(20000, 20000 + len(y_pred_proba_test))
   blend = pd.DataFrame({'pred': y_pred_proba_test}, index=sub_ids)
   blend.to_csv('blend.csv')


@PipelineDecorator.pipeline(name='TPS', project='TPS', version='0.0.1')
def executing_pipeline(parameters):
   from pathlib import Path

   # Инициализируйте объект Task
   task = Task.init(project_name="TPS", task_name="TPS_pipeline_v8")

   # Получите датасет по его ID
   dataset_id = "61a444098f324650a669793d884b2f17"
   dataset = Dataset.get(dataset_id)
   path = Path(dataset.get_local_copy())


   parameters = task.connect(parameters)
   logger = task.get_logger()

   lib()
   labels, subs, sub_ids = step_one(path)
   ensemble_data = step_two(path, subs)
   X_train, X_val, y_train, y_val = step_three(ensemble_data, labels)
   y_pred_proba_val, y_pred_proba_test = step_four(X_train, X_val, y_train, ensemble_data)
   step_five(y_val, y_pred_proba_val)
   step_six(y_pred_proba_test, sub_ids)


if __name__ == '__main__':
   PipelineDecorator.run_locally()
   executing_pipeline(parameters)
   PipelineDecorator().stop()