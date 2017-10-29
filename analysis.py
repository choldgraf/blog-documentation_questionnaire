import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

sns.set(font_scale=1.5, style='white')
plt.ion()


# --- Read in data ---
data = pd.read_csv('./data/credit_enjoyment.csv', index_col=None)
data['docs-diff'] = data['docs-usual'] - data['docs-should']
data = data.sort_values(['docs-usual', 'docs-diff'], ascending=[True, False])
contribs = pd.read_csv('./data/contribs.csv', index_col=None)
contribs = contribs.loc[:, contribs.sum(0) > 2]  # Remove "other" responses


# --- Functions ---
def bootstrap_mean(dist, percentiles, n_boots=1000):
    # Bootstrap mean distribution
    boot_means = np.zeros(n_boots)
    ixs = np.random.randint(0, len(dist), len(dist) * n_boots).reshape(n_boots, -1)
    for ii, iix in enumerate(ixs):
        boot_means[ii] = np.mean(dist[iix])
    clo, chi = np.percentile(boot_means, percentiles)
    return clo, chi


# --- Figures ---
def plot_docs_usual_should(ax=None):
    sns.set(font_scale=2, style='white')
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("hls", 8)
    alpha = .5
    ms = 100
    for ii, (ix, row) in enumerate(data.iterrows()):
        color = colors[ii % 8]
        should = ax.scatter(ii, row['docs-should'], marker='o',
                            s=ms, c='k')
        usual = ax.scatter(ii, row['docs-usual'], marker='o', s=ms,
                           c='w', edgecolor='k', lw=1)
    ax.set(ylabel='Percent time', xlabel="Participant ID")
    should = Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=20)
    usual = Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='k',
                   markeredgewidth=1, markersize=20)
    ax.legend([should, usual],
              ['% time one should\nspend on docs', '% time one usually\nspends on docs'],
              loc=(1.02, .5))
    return ax
    
def plot_docs_diff_compare(ax=None):
    sns.set(font_scale=2, style='white')
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    col1 = plt.cm.coolwarm(0.)
    col2 = plt.cm.coolwarm(1.)
    ax.bar(range(len(data)), data['docs-diff'])
    for bar in ax.patches:
        height = bar.get_height()
        color = col2 if height < 0 else col1
        bar.set_color(color)
    ax.set(ylim=[-60, 60], xlabel="Participant ID", ylabel='Difference between\n"usual" and "should"')
    return fig

def plot_diff_hist(ax=None, cmap=plt.cm.coolwarm_r):
    sns.set(font_scale=1.2, style='white')
    clo, chi = bootstrap_mean(data['docs-diff'].dropna().values, [2.5, 97.5])
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Difference plot
    bins = np.arange(-50, 50, 10)

    counts, bins = np.histogram(data['docs-diff'].dropna(), bins=bins)
    colors = cmap(plt.Normalize(-50, 50)(bins[:-1]))
    ax.bar(bins[:-1], counts, color=colors, width=8)
    ax.plot([clo, chi],[np.max(counts) + 2] * 2, lw=10, color='k')
    ax.axvline(0, ls='--', c='k', alpha=.5)
    ax.annotate("Thinks they spend the\n right amount of time\non documentation",
                (0, 15), (.5, 19), arrowprops={'arrowstyle': '->', 'connectionstyle': 'angle3', 'linewidth': 3})
    ax.set(title="Difference in particiant views on\ntime spent on documentation",
           xticks=np.arange(-100, 100, 20),
           xlabel="Difference in % time (usual - should)",
           ylabel="Number of participants",
           xlim=[-60, 60], ylim=[0, 30])
    return ax


def plot_contrib_type_bar(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    sns.set(font_scale=1.5, style='white')
    tmp_contribs = contribs.sum(0)
    new_cols = []
    for col in tmp_contribs.index:
        words = col.split(' ')
        for ii in [12, 8, 4]:
            if ii < len(words):
                words.insert(ii, '\n')
        new_cols.append(' '.join(words))
    tmp_contribs.index = new_cols
    ax = tmp_contribs.sort_values().plot.bar()
    ax.set(ylabel="Number of yes responses",
           title="Open Source activities of SciPy attendees")
    plt.setp(ax.get_xticklabels(), rotation=45,
             horizontalalignment='right')
    return ax

def plot_credit_enjoyment(ax=None):
    # Tidy the data
    new_columns = []
    keep_cols = []
    for col in data.columns:
        if not any(ii in col for ii in ['enjoyment', 'credit']):
            continue

        keep_cols.append(col)
        kind, col = col.split('-')

        col = col.replace(kind + '_', '')
        replace = [('manage_comm', 'managing_comm')]
        for first, second in replace:
            col = col.replace(first, second)
        new_columns.append([col, kind])

    df_new = data[keep_cols]
    df_new.columns = pd.MultiIndex.from_tuples(new_columns, names=['question', 'kind'])
    df_new.index.name = 'id'
    df_new = df_new.stack(['question', 'kind']).to_frame('value').reset_index(['question', 'kind'])
    df_new = df_new.sort_values(['question', 'value'], ascending=False)
    sorted_questions = df_new.query('kind == "credit"').groupby('question').mean().sort_values('value', ascending=False).index.values

    # Generate the plot
    label_mapping = {'fixing_bugs': 'Fixing Bugs', 'infrastructure': 'Infrastructure / Build Systems',
                     'managing_communities': 'Managing Communities', 'responding_to_issues': 'Responding to Issues',
                     'reviewing_code': 'Reviewing Code', 'reviewing_documentation': 'Reviewing Documentation',
                     'writing_code': 'Writing Code', 'writing_documentation': 'Writing Documentation'}
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='question', y='value', hue='kind', order=sorted_questions, data=df_new, palette=plt.cm.tab10([0., .1]))
    ax.legend(np.array(ax.patches)[[0, -1]], ['Perceived\nCredit', 'Enjoyment'], loc=(1.05, .85))
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right');
    ax.set_xticklabels([label_mapping[lab.get_text()] for lab in ax.get_xticklabels()]);
    ax.set(ylabel='Ordinal Response\n(1: not at all, 5: a lot)', xlabel='Task',
           title="Enjoyment vs. Perceived Credit\nReceived for Open-Source Tasks")
    return ax