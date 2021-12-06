import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from itertools import chain
import pickle
import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def main():

    YEAR = 2015
    NUMBER_OF_YEAR = 7

    dataframe_path = "test/interim/dataframe.pkl"
    processed_data_path =  "test/interim/all_docs.pkl"
    models_path = "test/processed/models.pkl"
    results_path = "test/processed/results.pkl"
    names_path = "test/processed/names.pkl"
    authors_path = "test/interim/authors.pkl"
    missing_author_years_path = "test/interim/missing_author_years.pkl"
    number_of_authors = 3
    raw_data_path = "test/testdata/test_data.csv"
    topics_range = [15]
    folder = "test"
    org_data = "test_data_new.csv"

    data = pickle.load(open(dataframe_path, 'rb'))

    all_docs = pickle.load(open(processed_data_path, 'rb'))

    models = pickle.load(open(models_path, 'rb'))
    results = pickle.load(open(results_path, 'rb'))
    names = pickle.load(open(names_path, 'rb'))

    authors = pickle.load(open(authors_path, 'rb'))
    missing_author_years = pickle.load(open(missing_author_years_path, 'rb'))

    # model_diff = topics_range[1] - topics_range[0]

    num_topics_list = [topics_range[0]] # , topics_range[-1]+model_diff, model_diff)

    topicnames = {
        num_topics : ["Topic" + str(i) for i in range(num_topics)] for num_topics in num_topics_list
    }

    print(topicnames)

    # index names
    docnames = ["Doc" + str(i) for i in range(len(all_docs))]

    # Make the pandas dataframe
    df_document_topic = {
        num_topics : pd.DataFrame(results[f'{num_topics}'], columns=topicnames[num_topics], index=docnames) for num_topics in num_topics_list
    }

    # Get dominant topic for each document
    dominant_topic = {
        num_topics : np.argmax(df_document_topic[num_topics].values, axis=1) for num_topics in num_topics_list
    }

    for num_topics, df in df_document_topic.items():
        df['dominant_topic'] = dominant_topic[num_topics]
            



    author_list = []
    year_list = []
    for author in authors.keys():
        for i in range(NUMBER_OF_YEAR):
            if (YEAR + i) not in missing_author_years[author]:
                author_list.append(author)
                year_list.append(YEAR + i)

    for df in df_document_topic.values():
        df['author'] = author_list
        df['year'] = year_list



    averaged = {
        num_topics : df_document_topic[num_topics].groupby('author').mean().drop(['dominant_topic', 'year'], axis=1) for num_topics in df_document_topic.keys()
    }

    filtered = {
        threshold : {num_topics : averaged[num_topics].mask(averaged[num_topics] < threshold, other=0) for num_topics in averaged.keys()} for threshold in [.1]
    }


    labels = {}
    for num_topics in topics_range:
        labels[num_topics] = filtered[.1][num_topics].index.to_list()
        labels[num_topics].extend(filtered[.1][num_topics].columns.to_list())


    sources = {threshold : {} for threshold in [.1]}
    targets = {threshold : {} for threshold in [.1]}
    values = {threshold : {} for threshold in [.1]}

    for threshold in [.1]:
        for num_topics in topics_range:
            curr_sources = []
            curr_targets = []
            curr_values = []
            index_counter = 0
            for index, row in filtered[threshold][num_topics].iterrows():
                for i, value in enumerate(row):
                    if value != 0:
                        curr_sources.append(index_counter)
                        curr_targets.append(number_of_authors + i)
                        curr_values.append(value)
                index_counter += 1
            sources[threshold][num_topics] = curr_sources
            targets[threshold][num_topics] = curr_targets
            values[threshold][num_topics] = curr_values

    positions = {
        num_topics : {label : i for i, label in enumerate(labels[num_topics])} for num_topics in averaged.keys()
    }
    
    print('sources, targets, and values for sankey DONE')

    def split_into_ranks(array):
        ranks = []
        for value in array:
            for i, percentage in enumerate(np.arange(.1, 1.1, .1)):
                if value <= np.quantile(array, percentage):
                    ranks.append(i + 1)
                    break
        return ranks

    final_values = {threshold : {} for threshold in [.1]}

    for threshold in [.1]:
        for num_topics in topics_range:
            curr_values_array = np.array(values[threshold][num_topics])
            final_values[threshold][num_topics] = split_into_ranks(curr_values_array)


    counts = CountVectorizer().fit_transform(data['abstract_processed'])
    transformed_list = []
    for model in models.values():
        transformed_list.append(model.transform(counts))


    dataframes = {threshold : {} for threshold in [.1]}
    for i, matrix in enumerate(transformed_list):
        for threshold in [.1]:
            df = pd.DataFrame(matrix)
            df.mask(df < threshold, other=0, inplace=True)
            df['HDSI_author'] = data['HDSI_author']
            df['year'] = data['year']
            df['citations'] = data['times_cited'] + 1

            # normalization of citations: Scaling to a range [0, 1]
            df['citations_norm'] = df.groupby(by=['HDSI_author', 'year'])['citations'].apply(lambda x: (x-x.min())/(x.max()-x.min())) #normalize_by_group(df=df, by=['author', 'year'])['citations']
            df['abstract'] = data['abstract']
            df['title'] = data['title']
            df.fillna(1, inplace=True)
            
            #alpha weight parameter for weighting importance of citations vs topic relation
            alpha = .75
            for topic_num in range(15):
                df[f'{topic_num}_relevance'] = alpha * df[topic_num] + (1-alpha) * df['citations_norm']
            dataframes[threshold][15] = df

    def create_top_list(data_frame, num_topics, threshold):
        top_5s = []
        the_filter = filtered[threshold][num_topics]
        for topic in range(num_topics):
            relevant = the_filter[the_filter[f'Topic{topic}'] != 0].index.to_list()
    #         print(relevant)
            to_append = data_frame[data_frame[f'{topic}_relevance'] > 0].reset_index()
            #   print(to_append.columns)
            to_append = to_append[to_append['HDSI_author'].isin(relevant)].reset_index()
            top_5s.append(to_append)
        return top_5s
        
    print(dataframes)

    tops = {
        threshold : {num_topics : create_top_list(dataframes[threshold][num_topics], num_topics, threshold) for num_topics in num_topics_list} for threshold in [.1]
    }

    print('large dataframe including the document-topic and calculated relevant score DONE')

    # sankey diagrams for diff numbers of topics
    def display_topics_list(model, feature_names, no_top_words):
        topic_list = []
        for topic_idx, topic in enumerate(model.components_):
            topic_list.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        return topic_list

    link_labels = {}
    for num_topics in topics_range:
        link_labels[num_topics] = labels[num_topics].copy()
        link_labels[num_topics][number_of_authors:] = display_topics_list(models[f'{num_topics}'], names, 10)

    lst_of_topics = topics_range.copy()

    heights = {
        lst_of_topics[0] : 1000,
        # lst_of_topics[1] : 1500,
        # lst_of_topics[2] : 2000,
        # lst_of_topics[3] : 2500,
        # lst_of_topics[4] : 3000
    }

    figs = {threshold : {} for threshold in [.1]}
    for threshold in [.1]:
        for num_topics in topics_range:
            fig = go.Figure(data=[go.Sankey(
                node = dict(
                    pad = 15,
                    thickness = 20,
                    line = dict(color = 'black', width = 0.5),
                    label = labels[num_topics],
                    color = ['#666699' for i in range(len(labels[num_topics]))],
                    customdata = link_labels[num_topics],
                    hovertemplate='%{customdata} Total Flow: %{value}<extra></extra>'
                ),
                link = dict(
                    color = ['rgba(204, 204, 204, .5)' for i in range(len(sources[threshold][num_topics]))],
                    source = sources[threshold][num_topics],
                    target = targets[threshold][num_topics],
                    value = final_values[threshold][num_topics]
                )
            )])
            fig.update_layout(title_text="Author Topic Connections", font=dict(size = 10, color = 'white'), height=heights[num_topics], paper_bgcolor="black", plot_bgcolor='black')
            figs[threshold][num_topics] = fig


    top_words = {
        lst_of_topics[0] : display_topics_list(models['{}'.format(lst_of_topics[0])], names, 10),
        # lst_of_topics[1] : display_topics_list(models['{}'.format(lst_of_topics[1])], names, 10),
        # lst_of_topics[2] : display_topics_list(models['{}'.format(lst_of_topics[2])], names, 10),
        # lst_of_topics[3] : display_topics_list(models['{}'.format(lst_of_topics[3])], names, 10),
        # lst_of_topics[4] : display_topics_list(models['{}'.format(lst_of_topics[4])], names, 10)
    }

    # 'final_hdsi_faculty_updated.csv'
    combined = pd.read_csv(raw_data_path)
    # combined[combined.title == 'Elder-Rule-Staircodes for Augmented Metric Spaces'].abstract

    locations = {}
    for i, word in enumerate(names):
        locations[word] = i

    print('sankey diagram for different numbers of topics DONE')

    pickle.dump(figs, open('{}/sankey_dash/figs.pkl'.format(folder), 'wb'))
    pickle.dump(tops, open('{}/sankey_dash/tops.pkl'.format(folder), 'wb'))
    pickle.dump(top_words, open('{}/sankey_dash/top_words.pkl'.format(folder), 'wb'))
    pickle.dump(author_list, open('{}/sankey_dash/author_list.pkl'.format(folder), 'wb'))
    pickle.dump(labels, open('{}/sankey_dash/labels.pkl'.format(folder), 'wb'))
    pickle.dump(positions, open('{}/sankey_dash/positions.pkl'.format(folder), 'wb'))
    pickle.dump(sources, open('{}/sankey_dash/sources.pkl'.format(folder), 'wb'))
    pickle.dump(targets, open('{}/sankey_dash/targets.pkl'.format(folder), 'wb'))
    
    pickle.dump(locations, open('{}/sankey_dash/locations.pkl'.format(folder), 'wb'))
    pickle.dump(models, open('{}/sankey_dash/models.pkl'.format(folder), 'wb'))
    pickle.dump(names, open('{}/sankey_dash/names.pkl'.format(folder), 'wb'))

    combined = pd.read_csv('{}/testdata/{}'.format(folder, "test_data.csv"))
    pickle.dump(combined, open('{}/sankey_dash/combined.pkl'.format(folder), 'wb'))

    figs = pickle.load(open('{}/sankey_dash/figs.pkl'.format(folder), 'rb'))
    tops = pickle.load(open('{}/sankey_dash/tops.pkl'.format(folder), 'rb'))
    top_words = pickle.load(open('{}/sankey_dash/top_words.pkl'.format(folder), 'rb'))
    combined = pickle.load(open('{}/sankey_dash/combined.pkl'.format(folder), 'rb'))
    author_list = pickle.load(open('{}/sankey_dash/author_list.pkl'.format(folder), 'rb'))
    labels = pickle.load(open('{}/sankey_dash/labels.pkl'.format(folder), 'rb'))
    positions = pickle.load(open('{}/sankey_dash/positions.pkl'.format(folder), 'rb'))
    sources = pickle.load(open('{}/sankey_dash/sources.pkl'.format(folder), 'rb'))
    targets = pickle.load(open('{}/sankey_dash/targets.pkl'.format(folder), 'rb'))
    locations = pickle.load(open('{}/sankey_dash/locations.pkl'.format(folder), 'rb'))
    models = pickle.load(open('{}/sankey_dash/models.pkl'.format(folder), 'rb'))
    names = pickle.load(open('{}/sankey_dash/names.pkl'.format(folder), 'rb'))

    threshold = .1
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    topics_range = json.load(open('config/viz-params.json', 'r'))['topics_range']

    app.layout = html.Div([
        dbc.Row([
            dcc.Dropdown(
            id='graph-dropdown',
            placeholder='select number of LDA topics',
            options=[{'label' : f'{i} Topic Model', 'value' : i} for i in topics_range],
            style={
                'color' : 'black',
                'background-color' : '#666699',
                'width' : '200%',
                'align-items' : 'left',
                'justify-content' : 'left',
                'padding-left' : '15px'
            },
            value=10
            )
        ]),
        dbc.Row([
        dbc.Col(html.Div([
            dcc.Graph(
            id = 'graph',
            figure = figs[.1][15]
            )
            ],
            style={
            'height' : '100vh',
            'overflow-y' : 'scroll'
            }
        )
        ),
            dbc.Col(html.Div([dbc.Col([
            dcc.Dropdown(
                id='dropdown_menu',
                placeholder='Select a topic',
                options=[{'label' : f'Topic {topic}: {top_words[15][topic]}', 'value' : topic} for topic in range(15)],
                style={
                'color' : 'black',
                'background-color' : 'white'
                }
            ),
            dcc.Dropdown(
                id='researcher-dropdown',
                placeholder='Select Researchers',
                options=[{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in set(author_list)],
                style={
                'color' : 'black',
                'background-color' : 'white'
                }
            )]),
            dbc.Col(
                dcc.Dropdown(
                id='word-search',
                placeholder='Search by word',
                options=[{'label' : word, 'value' : word} for word in names],
                style={
                    'color' : 'black',
                    'background-color' : 'white'
                },
                value=[],
                multi=True
                )
            ),
            html.Div(
                id='paper_container', 
                children=[
                html.P(
                    children=['Top 5 Papers'],
                    id='titles_and_authors', 
                    draggable=False, 
                    style={
                    'font-size' :'150%',
                    'font-family' : 'Verdana'
                    }
                ),
                ],
            ),
            ], 
            style={
                'height' : '100vh',
                'overflow-y' : 'scroll'
            }
            )
            )
        ]
        )]
    )

    @app.callback(
        Output('titles_and_authors', 'children'),
        Output('researcher-dropdown', 'options'),
        Input('dropdown_menu', 'value'),
        Input('graph-dropdown', 'value'),
        Input('researcher-dropdown', 'value'),
        Input('word-search', 'value')
    )
    def update_p(topic, num_topics, author, words):
        if len(words) != 0:
            doc_vec = np.zeros((1, len(names)))
            for word in words:
                doc_vec[0][locations[word]] += 1
            relations = np.round(models[f'{num_topics}'].transform(doc_vec), 5).tolist()[0]
            pairs = [(i, relation) for i, relation in enumerate(relations)]
            pairs.sort(reverse=True, key=lambda x: x[1])
            to_return = [[html.Br(), f'Topic{pair[0]}: {pair[1]}', html.Br()] for pair in pairs]
            return list(chain(*to_return)), [{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in set(author_list)]

        if topic == None and author == None:
            return ['Make a selection'], [{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in set(author_list)]

        if topic != None and author == None:
            df = tops[threshold][num_topics][topic]
            # df_authors = df.HDSI_author.unique()
            max_vals = df.groupby('HDSI_author').max()[f'{topic}_relevance']

            to_return = [[f'{name}:', html.Br(), 
                f'{df[df[f"{topic}_relevance"] == max_vals.loc[name]]["title"].to_list()[0]}',
                html.Details([html.Summary('Abstract'),
                            html.Div(combined[combined.title == f'{df[df[f"{topic}_relevance"] == max_vals.loc[name]]["title"].to_list()[0]}'].abstract)],
                            style={
                                'font-size' :'80%',
                                'font-family' : 'Verdana'}),
                html.Br()] for i, name in enumerate(max_vals.index)]
            return list(chain(*to_return)), [{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in tops[threshold][num_topics][topic].HDSI_author.unique()]

        if topic == None and author != None:
            to_return = []
            for topic_num in range(num_topics):
                df = tops[threshold][num_topics][topic_num]
                if author in df.HDSI_author.unique():
                    max_vals = df.groupby('HDSI_author').max()[f'{topic_num}_relevance']
            
                    to_return.append([f'Topic {topic_num}:', html.Br(), 
                        f'{df[df[f"{topic_num}_relevance"] == max_vals.loc[author]]["title"].to_list()[0]}', 
                        html.Details([html.Summary('Abstract'), 
                                    html.Div(combined[combined.title == f'{df[df[f"{topic_num}_relevance"] == max_vals.loc[author]]["title"].to_list()[0]}'].abstract)],
                                    style={
                                        'font-size' :'80%',
                                        'font-family' : 'Verdana'},
                                    ),
                        html.Br()])
            return list(chain(*to_return)), [{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in set(author_list)]

        if topic != None and author != None:
            df = tops[threshold][num_topics][topic]
            df = df[df['HDSI_author'] == author]
            df.sort_values(by=f'{topic}_relevance', ascending=False, inplace=True)
            titles = df.head(10)['title'].to_list()
            
            to_return = [
                [f'{i} : {title}', 
                html.Details([html.Summary('Abstract'), 
                            html.Div(combined[combined.title == title].abstract)], 
                            style={
                                'font-size' :'80%',
                                'font-family' : 'Verdana'}), 
                html.Br()] for i, title in enumerate(titles)]
            return list(chain(*to_return)), [{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in tops[threshold][num_topics][topic].HDSI_author.unique()]
        


    @app.callback(
        [Output('graph', 'figure'), Output('dropdown_menu', 'options')],
        [Input('graph-dropdown', 'value'), Input('dropdown_menu', 'value'), Input('researcher-dropdown', 'value'), Input('word-search', 'value')],
        State('graph', 'figure')
    )

    def update_graph(value, topic, author, words, previous_fig):
        if len(previous_fig['data'][0]['node']['color']) != value + 50:
            figs[threshold][value].update_traces(node = dict(color = ['#666699' for i in range(len(labels[value]))]), link = dict(color = ['rgba(204, 204, 204, .5)' for i in range(len(sources[threshold][value]))]))
            return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]

        if len(words) != 0:
            doc_vec = np.zeros((1, len(names)))
            for word in words:
                doc_vec[0][locations[word]] += 1
            relations = np.round(models[f'{value}'].transform(doc_vec), 3).tolist()[0]
            opacity = {(i+50) : relation for i, relation in enumerate(relations) if relation > .1}
            node_colors = ['#666699' if (i not in opacity.keys()) else f'rgba(255, 255, 0, {opacity[i]})' for i in range(len(labels[value]))]
            valid_targets = [positions[value][f'Topic{i-50}'] for i in opacity.keys()]
            link_colors = ['rgba(204, 204, 204, .5)' if target not in valid_targets else f'rgba(255, 255, 0, .5)' for target in targets[threshold][value]]
            figs[threshold][value].update_traces(node = dict(color = node_colors), link = dict(color = link_colors)),
            return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]


        if topic == None and author == None:
            figs[threshold][value].update_traces(node = dict(color = ['#666699' for i in range(len(labels[value]))]), link = dict(color = ['rgba(204, 204, 204, .5)' for i in range(len(sources[threshold][value]))]))
            return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]
        
        if topic != None and author == None:
            node_colors = ['#666699' if (i != positions[value][f'Topic{topic}']) else '#ffff00' for i in range(len(labels[value]))]
            link_colors = ['rgba(204, 204, 204, .5)' if target != positions[value][f'Topic{topic}'] else 'rgba(255, 255, 0, .5)' for target in targets[threshold][value]]
            figs[threshold][value].update_traces(node = dict(color = node_colors), link = dict(color = link_colors))
            return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]

        if topic == None and author != None:
            node_colors = ['#666699' if (i != positions[value][author]) else '#ffff00' for i in range(len(labels[value]))]
            link_colors = ['rgba(204, 204, 204, .5)' if source != positions[value][author] else 'rgba(255, 255, 0, .5)' for source in sources[threshold][value]]
            figs[threshold][value].update_traces(node = dict(color = node_colors), link = dict(color = link_colors))
            return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]

        if topic != None and author != None:
            node_colors = ['#666699' if (i != positions[value][author] and i != positions[value][f'Topic{topic}']) else '#ffff00' for i in range(len(labels[value]))]
            link_colors = ['rgba(204, 204, 204, .5)' if (source != positions[value][author] or target != positions[value][f'Topic{topic}']) else 'rgba(255, 255, 0, .5)' for source, target in zip(sources[threshold][value], targets[threshold][value])]
            figs[threshold][value].update_traces(node = dict(color = node_colors), link = dict(color = link_colors))
            return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]

    @app.callback(
        Output('researcher-dropdown', 'value'),
        Input('dropdown_menu', 'value'),
        State('dropdown_menu', 'value')
    )

    def reset_author(topic, previous):
        if topic != previous:
            return None

    app.run_server()


if __name__ == "__main__":
    main()