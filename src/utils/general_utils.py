import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from collections import Counter
import random
import torch
from sklearn.model_selection import train_test_split



#### PLOTS ####

def barplot_per_genre_over_years(MOVIES, genres, year_interval, variable, title):
    """
    Creates barplot for each genre grouped in intervals of year_interval for some variable
    """
    fig, axes = plt.subplots(10, 4, figsize=(30, 40), sharey = True)
    fig.delaxes(axes[9, 3])

    #colors
    colors = sns.color_palette("tab20", len(genres))
    color_iter = iter(colors)

    for j, genre in enumerate(genres):
        # Make bins for interval of years to plot as bars
        df_filtered = MOVIES[MOVIES["new_genres"].apply(lambda x: genre in x)]
        bin_size = year_interval
        movies_year_runtime = df_filtered.groupby("release_year")[variable].median()
        binned_counts = {}
        for i in range(int(movies_year_runtime.index.min()), int(movies_year_runtime.index.max()), bin_size):
            bin_start, bin_end = i, i + bin_size - 1
            
            # Sum all the runtimes counts in this range.
            total_in_bin = movies_year_runtime[(movies_year_runtime.index >= bin_start) & (movies_year_runtime.index <= bin_end)].mean()
            
            binned_counts[f'{bin_start}-{bin_end}'] = total_in_bin


        binned_token_counts = pd.Series(binned_counts)
        color = next(color_iter)
        ax = axes[j//4, j%4]
        binned_token_counts.plot(kind='bar', ax=ax, color=color, fontsize=15)
        ax.set_title(genre, size = 20)

    fig.supxlabel('Release year', size = 25)
    fig.supylabel(f'{variable.capitalize()}', size = 25)
    fig.suptitle(title, size = 40, y=1.01)
    fig.tight_layout(rect=(0.025,0.025,1,1))


def barplot_means_per_genre(MOVIES, genres, variable, title, median=True, zeros=True):
    """
    Creates barplot with mean value of variable for each genre.
    If median is True it takes the mean of medians over years.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    colors = sns.color_palette("tab20", len(genres))
    color_iter = iter(colors)

    average = []
    sem = []

    for idx, genre in enumerate(genres):

        df_filtered = MOVIES[MOVIES["new_genres"].apply(lambda x: genre in x)]
        if not zeros:
            df_filtered = df_filtered[df_filtered[variable] != 0] #remove zeros if zeros is False
        if median: # If median is true, take the median over year
            group = df_filtered.groupby("release_year")[variable].median()
        else:
            group = df_filtered[variable]
        average.append(group.mean())
        sem.append(group.std()/np.sqrt(group.size))
        
    ax.tick_params(axis='x', labelrotation=90)

    for i in range(len(genres)):
        color = next(color_iter)
        ax.bar(genres[i], average[i], color=color)
        ax.errorbar(genres[i], average[i], yerr=sem[i],capsize=4,  color = 'k')
        
    ax.set_xlim(-0.5, len(genres) - 0.5) # Set limits to avoid extra space on sides
    plt.tight_layout()
    ax.set_xlabel("Genre")
    ax.set_ylabel(variable.title())
    ax.set_title(title)


def scatter_plot_per_genre(MOVIES, genre, variablex, variabley, title, zeros=True):
    """
    Creates scatter plot with regression line for each genre.
    """
    
    fig, axes = plt.subplots(10, 4, figsize=(30, 40), sharey = True)
    fig.delaxes(axes[9, 3])

    colors = sns.color_palette("tab20", len(genre))
    color_iter = iter(colors)

    for j, genre in enumerate(genre):

        df_filtered = MOVIES[MOVIES["new_genres"].apply(lambda x: genre in x)]
        if not zeros:
            df_filtered = df_filtered[df_filtered[variablex] != 0] #e.g. No movies has 0 in budget, so we remove them form the analysis
            df_filtered = df_filtered[df_filtered[variabley] != 0] # e.g. remove 0 revenue movie, as the lowest recorded box office is 11$ according to Wikipedia
        color = next(color_iter)
        ax = axes[j//4, j%4]
        sns.regplot(data=df_filtered, x=variablex, y=variabley, color=color, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.set_title(genre, size = 20)
    fig.supxlabel(variablex.title(), size = 25)
    fig.supylabel(variabley.title(), size = 25)
    fig.suptitle(title, size = 40, y=1.01)
    fig.tight_layout(rect=(0.025,0.025,1,1))


def heatmap_per_genre(MOVIES, genres, varx, vary, varz, stepsizex, stepsizey, title):
    """
    Creates heatmap of 3 variables, where number of cells is decided by stepsizes
    """

    fig, axes = plt.subplots(10, 4, figsize=(30, 40))
    fig.delaxes(axes[9, 3])

    for j, genre in enumerate(genres):
        df_filtered = MOVIES[MOVIES["new_genres"].apply(lambda x: genre in x)]
        df_filtered[f'interval_y'] = df_filtered[vary].astype(float).apply(lambda x: f'{int(x/stepsizey)*stepsizey:.1f}-{int(x/stepsizey+1)*stepsizey:.1f}')
        df_filtered[f'interval_x'] = df_filtered[varx].astype(float).apply(lambda x: f'{int(x/stepsizex)*stepsizex:.1f}-{int(x/stepsizex+1)*stepsizex:.1f}')
        df_temp = pd.crosstab(df_filtered['interval_y'], df_filtered['interval_x'], values = df_filtered[varz],
                    margins=False, aggfunc='median')

        ax = axes[j//4, j%4]
        sns.heatmap(df_temp, annot=False, ax=ax, cmap='viridis')
        ax.invert_yaxis()
        ax.set_title(genre, size = 20)
        

    fig.supxlabel(varx.title(), size = 25)
    fig.supylabel(vary.title(), size = 25)
    fig.suptitle(title, size = 40, y=1.01)
    fig.tight_layout(rect=(0.025,0.025,1,1))


def bar_per_genre(MOVIES, genres, column_names, title, xlabel, ylabel, legend_title, stacked=False):
    """
    Bar plot of some means of variables. If stacked is True return stacked barplot, 
    else creates normal or grouped barplot.
    """
    plot_df = pd.DataFrame(columns=column_names  + ['genre'])
    for _, genre in enumerate(genres):
        filtered_df = MOVIES[MOVIES["new_genres"].apply(lambda x: genre in x)]
        plot_df.loc[len(plot_df)] = filtered_df[column_names].mean()
        
    plot_df["genre"] = genres
    plot_df.set_index('genre', inplace=True)    

    if stacked:
        plot_df.plot(kind='bar', stacked=True, figsize=(14, 8))
    else:
        plot_df.plot(kind='bar', figsize=(14, 8))

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(title=legend_title, fontsize=12)

    plt.tight_layout()
    plt.show()


def lineplot_per_genre_over_years(MOVIES,  variables, title, xlabel, ylabel, legend_title, year_interval=10):
    """
    Creates line plot with one or more variables.
    """
    span_of_years = year_interval
    MOVIES['release_decade'] = (MOVIES['release_year'] // span_of_years) * span_of_years

    average_actors_by_decade = MOVIES.groupby('release_decade')[variables].mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    average_actors_by_decade.plot(ax=ax)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(legend_title, fontsize=12)

    plt.tight_layout()
    plt.show()
