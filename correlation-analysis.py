import pandas
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_analytics(path_to_file):

    df = pandas.read_excel(path_to_file)

    path_to_file = path_to_file.replace('.xlsx', '')

    h_labels = [x.replace('_', ' ').title() for x in 
                list(df.select_dtypes(include=['number', 'bool']).columns.values)]

    correlation = df.corr()

    fig, ax = plt.subplots(figsize=(10,6))
    _ = sns.heatmap(correlation, annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)

    correlation.unstack().sort_values(ascending=False).drop_duplicates().to_excel(f'{path_to_file}_more_correlated.xlsx')
    df.describe().to_excel(f'{path_to_file}_describe.xlsx')
    fig.savefig('{path_to_file}_corr.png'.format(path_to_file=path_to_file))

def main():
    path_to_file = input('Enter the path to the xlsx file with metrics: ')
    calculate_analytics(path_to_file)

if __name__ == '__main__':
    main()