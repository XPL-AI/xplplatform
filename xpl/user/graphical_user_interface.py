import os
import pandas
import visdom
import random

#from sklearn.gaussian_process import GaussianProcessRegressor


class GraphicalUserInterface:

    def __init__(self):

        os.system('python -m visdom  > /dev/null 2>&1 &')
        self.__vis = visdom.Visdom()
        self.__color_dict = {}

    def plot_graph(self,
                   epoch_results: pandas.DataFrame,
                   best_results: pandas.DataFrame):
        lines = {}
        epoch_set_results = {}
        column_names = epoch_results.columns

        flat_epoch_results = epoch_results.reset_index().set_index('iteration')
        unique_input_set_names = flat_epoch_results['input_set'].unique()
        for input_set in unique_input_set_names:
            epoch_set_results[input_set] = flat_epoch_results[flat_epoch_results['input_set'] == input_set]
            for column_name in column_names:
                measurement_type = column_name.split('.')[1]
                graph_name = column_name.split('.')[2]
                line = self.__get_line_data_dict(input_set=input_set,
                                                 column_name=column_name,
                                                 epoch_results=epoch_set_results)
                if measurement_type not in lines:
                    lines[measurement_type] = []
                lines[measurement_type] += line

        for measurement_type in lines.keys():
            lines[measurement_type] += self.__get_best_result_dict(measurement_type=measurement_type,
                                                                   best_results=best_results)

            self.__vis._send({'data': lines[measurement_type],
                              'layout': self.__get_line_layout(measurement_type),
                              'win': measurement_type,
                              })
        return

    def __get_line_layout(self,
                          measurement_type: str):
        return {'title': measurement_type,
                'xaxis': {'title': 'Iteration',
                          'type': 'log'
                          },
                'yaxis': {'title': measurement_type,
                          'type': 'log'},
                }

    def __get_line_data_dict(self,
                             input_set: str,
                             column_name: str,
                             epoch_results: pandas.DataFrame):
        series = epoch_results[input_set][column_name].dropna()
        if series.empty:
            return {}
        X = series.index.values
        Y = series.values
        concept = column_name.split('.')[0]

        # gpr = IsotonicRegression()
        # gpr.fit(numpy.log2(X+1e-18).reshape(-1, 1), numpy.log2(Y + 1e-18).reshape(-1, 1))

        # sampled_x = numpy.linspace(start=numpy.log2(X[1]),
        #                            stop=numpy.log2(X[-1]),
        #                            num=len(X)).reshape(-1, 1)

        # sampled_y = gpr.predict(sampled_x)

        line_color, marker_color, dash = self.__get_line_color(concept, input_set)
        line = dict(  # x=(2**sampled_x).reshape(-1).tolist(),
            # y=(2**sampled_y).reshape(-1).tolist(),
            x=X.tolist(),
            y=Y.tolist(),
            mode='lines',
            type='custom',
            line={'color': line_color,
                  'width': '2',
                  'dash': dash,
                  },
            name=concept,
            showlegend=dash != 'dash')
        marker = dict(x=X.tolist(),
                      y=Y.tolist(),
                      mode='markers',
                      type='custom',
                      marker={'color': marker_color,
                              'symbol': 100,
                              'size': '1',
                              },
                      name=concept,
                      showlegend=False)

        return [line, marker]

    def __get_best_result_dict(self,
                               measurement_type: str,
                               best_results: pandas.DataFrame):

        X = []
        Y = []
        names = []
        for v in best_results.index:
            if v.find(f'.{measurement_type}.') > 0:
                X.append(best_results['iteration'][v])
                Y.append(best_results['measurements'][v])
                names.append(f'{v}\n{best_results["model_id"][v]}')

        if len(X) == 0:
            return []
        return [dict(x=X,
                     y=Y,
                     mode='markers',
                     type='custom',
                     marker={'color': 'red',
                             'symbol': 101,
                             'size': '5',
                             },
                     text=names,
                     showlegend=False)
                ]

    def __get_line_color(self, concept, input_set):

        if concept not in self.__color_dict:
            self.__color_dict[concept] = ''.join(['rgba(',
                                                  f'{random.randint(a=64,b=192)},',
                                                 f'{random.randint(a=64,b=192)},',
                                                  f'{random.randint(a=64,b=255)},',
                                                  f'0.4',
                                                  ')'
                                                  ])
        color = self.__color_dict[concept]
        line_color = color
        marker_color = color.replace('0.4)', '0.9)')
        dash = 'solid'

        if input_set.lower().startswith('train'):
            dash = 'dash'
            line_color = color.replace('0.4)', '0.2)')
            marker_color = color.replace('0.4)', '0.3)')

        return line_color, marker_color, dash


if __name__ == '__main__':
    storage_path = '../models/xpl_test/mnist_digit_recognition/'
    ui = GraphicalUserInterface(user_id='xpl_test',
                                experiment_id='mnist_digit_recognition')

    execution_uuid = 'execution_2021-06-12_15-06-43_1f34bbcf-e10f-4851-9ab7-2657d1070dd1'
    epoch_results = pandas.read_csv(os.path.join(storage_path, execution_uuid, 'measurements.csv'),
                                    index_col=['model_id', 'graph_name', 'input_set', 'iteration'])
    best_results = pandas.read_csv(os.path.join(storage_path, execution_uuid, 'best.csv'),
                                   index_col=0)
    print(best_results)

    # for i in range(len(epoch_results)):
    ui.plot_graph(epoch_results, best_results)
    print(execution_uuid)
