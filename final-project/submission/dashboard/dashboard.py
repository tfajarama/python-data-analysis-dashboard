import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import streamlit as st
from babel.numbers import format_currency
from pathlib import Path
sns.set(style='dark')


# Helper Function

def create_successful_orders_df(df):
    """
    Create a DataFrame containing only successful orders.
    This function removes canceled or unavailable orders from the given DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing all orders.

    Returns:
    DataFrame: A DataFrame containing only successful orders.
    """
    return df[~df['order_status'].isin(['canceled', 'unavailable'])]


def create_failed_orders_df(df):
    """
    Create a DataFrame containing only failed orders.
    This function filters the given DataFrame to include only orders that are canceled or unavailable.

    Parameters:
    df (DataFrame): The DataFrame containing all orders.

    Returns:
    DataFrame: A DataFrame containing only failed orders.
    """
    return df[df['order_status'].isin(['canceled', 'unavailable'])]


def create_daily_orders_df(df):
    """
    Create a DataFrame containing daily aggregated order data.
    This function aggregates total orders, successful orders, and failed orders on a daily basis.
    It also calculates the revenue gain from successful orders and the revenue loss from failed orders.

    Parameters:
    df (DataFrame): The DataFrame containing all orders.

    Returns:
    DataFrame: A DataFrame containing daily aggregated order data, including total orders, successful orders,
               failed orders, revenue gain, and revenue loss.
    """
    # Aggregate total orders
    daily_orders_df = df.resample(rule='D', on='order_purchase_timestamp').agg(
        order_count=("order_id", "nunique")
    ).reset_index()

    all_successful_orders = create_successful_orders_df(df)
    # Group by payment sequential, so the payment_value is not duplicated
    all_successful_payment = all_successful_orders.groupby(
        ['order_purchase_timestamp', 'order_id', 'payment_sequential']).agg({
        'payment_value': 'first'
    }).reset_index()
    # Aggregate succesful orders and revenue gain
    daily_successful_orders = all_successful_payment.resample(rule='D', on='order_purchase_timestamp').agg(
        successful_order_count=("order_id", "nunique"),
        revenue_gain=("payment_value", "sum")
    ).reset_index()

    all_failed_orders = create_failed_orders_df(df)
    # Group by payment sequential, so the payment_value is not duplicated
    all_failed_payment = all_failed_orders.groupby(
        ['order_purchase_timestamp', 'order_id', 'payment_sequential']).agg({
        'payment_value': 'first'
    }).reset_index()
    # Aggregate failed orders and revenue loss
    daily_failed_orders = all_failed_payment.resample(rule='D', on='order_purchase_timestamp').agg(
        failed_order_count=("order_id", "nunique"),
        revenue_loss=("payment_value", "sum")
    ).reset_index()

    # Merge total orders, successful and failed orders
    daily_orders_df = pd.merge(daily_orders_df, daily_successful_orders,
                                          on='order_purchase_timestamp', how='left')
    daily_orders_df = pd.merge(daily_orders_df, daily_failed_orders, on='order_purchase_timestamp',
                                          how='left')

    # Fill NaN values with 0
    daily_orders_df.fillna(
        {'successful_order_count': 0, 'revenue_gain': 0, 'failed_order_count': 0, 'revenue_loss': 0}, inplace=True)
    daily_orders_df['successful_order_count'] = daily_orders_df['successful_order_count'].astype(
        int)
    daily_orders_df['failed_order_count'] = daily_orders_df['failed_order_count'].astype(int)

    return daily_orders_df


def visualize_daily_orders(daily_orders_df):
    """
    Visualize daily aggregated order data using Streamlit and Matplotlib.
    This function displays metrics for total successful orders, total revenue gain, total failed orders,
    and total revenue loss. It also plots the daily count of successful orders over time.

    Parameters:
    daily_orders_df (DataFrame): The DataFrame containing daily aggregated order data.

    Returns:
    None
    """
    # Clear any previous figure
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(
        daily_orders_df["order_purchase_timestamp"],
        daily_orders_df["successful_order_count"],
        marker='o',
        linewidth=2,
        color="#90CAF9"
    )
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=15)

    st.pyplot(fig)


def create_category_summary_df(df):
    """
    Create a summary DataFrame for product categories.
    This function aggregates successful orders by product category, calculating the total number of orders
    and the total revenue for each category.

    Parameters:
    df (DataFrame): The DataFrame containing all orders.

    Returns:
    DataFrame: A DataFrame containing the total number of orders and total revenue for each product category.
    """
    all_successful_orders = create_successful_orders_df(df)
    # Group by payment sequential, so the payment_value is not duplicated
    category_summary_df = all_successful_orders.groupby(
        ['product_category_name_english', 'order_id', 'payment_sequential']).agg(
        total_orders=('order_id', 'nunique'),
        total_revenue=('payment_value', 'first')
    ).reset_index()

    category_summary_df = category_summary_df.groupby(['product_category_name_english']).agg(
        total_orders=('order_id', 'nunique'),
        total_revenue=('total_revenue', 'sum')
    ).reset_index()

    return category_summary_df


def visualize_product_performance(category_summary_df):
    """
    Visualize the performance of product categories using Streamlit and Matplotlib.
    This function creates bar charts to display the best and worst performing product categories
    based on the number of sales and revenue gain. It shows the top 5 and bottom 5 categories
    for both metrics.

    Parameters:
    category_summary_df (DataFrame): The DataFrame containing the summary of product categories,
                                     including total orders and total revenue.

    Returns:
    None
    """
    # Clear any previous figure
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))

    colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

    # Best Performing Product
    sns.barplot(x="total_orders", y="product_category_name_english",
                data=category_summary_df.sort_values(by='total_orders', ascending=False).head(5),
                palette=colors, ax=ax[0], hue="product_category_name_english", legend=False)
    ax[0].set_ylabel(None)
    ax[0].set_xlabel("Number of Sales", fontsize=30)
    ax[0].set_title("Best Performing Product", loc="center", fontsize=50)
    ax[0].tick_params(axis='y', labelsize=35)
    ax[0].tick_params(axis='x', labelsize=30)

    # Worst Performing Product
    sns.barplot(x="total_orders", y="product_category_name_english",
                data=category_summary_df.sort_values(by='total_orders', ascending=True).head(5),
                palette=colors, ax=ax[1], hue="product_category_name_english", legend=False)
    ax[1].set_ylabel(None)
    ax[1].set_xlabel("Number of Sales", fontsize=30)
    ax[1].invert_xaxis()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_title("Worst Performing Product", loc="center", fontsize=50)
    ax[1].tick_params(axis='y', labelsize=35)
    ax[1].tick_params(axis='x', labelsize=30)

    st.pyplot(fig)

    st.subheader("Best & Worst Performing Product by Revenue Gain Nominal")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))

    colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

    # Best Performing Product
    sns.barplot(x="total_revenue", y="product_category_name_english",
                data=category_summary_df.sort_values(by='total_revenue', ascending=False).head(5),
                palette=colors, ax=ax[0], hue="product_category_name_english", legend=False)
    ax[0].set_ylabel(None)
    ax[0].set_xlabel("Revenue Gain Nominal (in multiplies of BRL1.000.000,00)", fontsize=30)
    ax[0].set_title("Best Performing Product", loc="center", fontsize=50)
    ax[0].tick_params(axis='y', labelsize=35)
    ax[0].tick_params(axis='x', labelsize=30)

    # Worst Performing Product
    sns.barplot(x="total_revenue", y="product_category_name_english",
                data=category_summary_df.sort_values(by='total_revenue', ascending=True).head(5),
                palette=colors, ax=ax[1], hue="product_category_name_english", legend=False)
    ax[1].set_ylabel(None)
    ax[1].set_xlabel("Revenue Gain Nominal (in BRL)", fontsize=30)
    ax[1].invert_xaxis()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_title("Worst Performing Product", loc="center", fontsize=50)
    ax[1].tick_params(axis='y', labelsize=35)
    ax[1].tick_params(axis='x', labelsize=30)

    st.pyplot(fig)


def create_delivery_summary_df(df):
    """
    Create a summary DataFrame for order deliveries.
    This function filters successful orders to include only those that have been delivered,
    calculates the difference between the actual delivery date and the estimated delivery date,
    and aggregates the delivery information for each order.

    Parameters:
    df (DataFrame): The DataFrame containing all orders.

    Returns:
    DataFrame: A DataFrame containing the delivery summary for each order, including the delivered date,
               estimated delivery date, and the difference between the actual and estimated delivery dates.
    """
    all_successful_orders = create_successful_orders_df(df)

    all_delivered_orders = all_successful_orders.loc[
        (all_successful_orders['order_delivered_customer_date'].isna() == False)]

    all_delivered_orders = all_delivered_orders.copy()

    all_delivered_orders['delivery_estimation_difference'] = (
                all_delivered_orders['order_delivered_customer_date'] - all_delivered_orders[
            'order_estimated_delivery_date']).dt.days

    delivery_summary_df = all_delivered_orders.groupby('order_id').agg(
        delivered_date=('order_delivered_customer_date', 'first'),
        estimated_date=('order_estimated_delivery_date', 'first'),
        delivery_estimation_diff=('delivery_estimation_difference', 'first')
    ).reset_index()

    return delivery_summary_df


def visualize_delivery_performance(delivery_summary_df):
    """
    Visualize the delivery performance using Streamlit and Matplotlib.
    This function creates a histogram to show the difference in days between the actual delivery date
    and the estimated delivery date for orders. It also adds a normal distribution curve to the histogram.

    Parameters:
    delivery_summary_df (DataFrame): The DataFrame containing the delivery summary for each order,
                                     including the difference between the actual and estimated delivery dates.

    Returns:
    None
    """
    # Clear any previous figure
    plt.clf()
    plt.close()

    plt.figure(figsize=(15, 5))
    sns.histplot(delivery_summary_df['delivery_estimation_diff'], bins=30, color="#72BCD4")
    plt.title('Difference in Days Between Order Delivered to Customer and Estimated Delivery Date')
    plt.xlabel('Days Difference')
    plt.ylabel('Number of Orders')

    # Add a curve belt line
    mean = delivery_summary_df['delivery_estimation_diff'].mean()
    std = delivery_summary_df['delivery_estimation_diff'].std()
    x = np.linspace(delivery_summary_df['delivery_estimation_diff'].min(),
                    delivery_summary_df['delivery_estimation_diff'].max(), 100)
    p = norm.pdf(x, mean, std)
    plt.plot(x, p * len(delivery_summary_df['delivery_estimation_diff']) * (x[1] - x[0]), 'r--')

    st.pyplot(plt)


def create_shipment_summary_df(df):
    all_successful_orders = create_successful_orders_df(df)

    all_shipped_orders = all_successful_orders.loc[
        (all_successful_orders['order_delivered_carrier_date'].isna() == False)]

    all_shipped_orders = all_shipped_orders.copy()

    all_shipped_orders['shipping_limit_difference'] = (
                all_shipped_orders['order_delivered_carrier_date'] - all_shipped_orders['shipping_limit_date']).dt.days

    shipment_summary_df = all_shipped_orders.groupby('order_id').agg(
        shipped_date=('order_delivered_carrier_date', 'first'),
        limit_date=('shipping_limit_date', 'first'),
        shipment_limit_diff=('shipping_limit_difference', 'first')
    ).reset_index()

    return shipment_summary_df


def visualize_shipment_performance(shipment_summary_df):
    # Clear any previous figure
    plt.clf()
    plt.close()

    plt.figure(figsize=(15, 5))
    sns.histplot(shipment_summary_df['shipment_limit_diff'], bins=30, color="#72BCD4")
    plt.title('Difference in Days Between Order Shipped to Carrier and Shipping Limit Date')
    plt.xlabel('Days Difference')
    plt.ylabel('Number of Orders')

    # Add a curve belt line
    mean = shipment_summary_df['shipment_limit_diff'].mean()
    std = shipment_summary_df['shipment_limit_diff'].std()
    x = np.linspace(shipment_summary_df['shipment_limit_diff'].min(),
                    shipment_summary_df['shipment_limit_diff'].max(), 100)
    p = norm.pdf(x, mean, std)
    plt.plot(x, p * len(shipment_summary_df['shipment_limit_diff']) * (x[1] - x[0]), 'r--')

    st.pyplot(plt)


def create_bystate_df(df, is_seller=True):
    seller_or_customer = 'seller' if is_seller else 'customer'

    all_successful_orders = create_successful_orders_df(df)

    all_order_count_state = all_successful_orders.groupby(f'{seller_or_customer}_state').agg(
        order_count=('order_id', 'nunique')
    ).reset_index()

    all_geolocation_state = pd.DataFrame(['SP','RN','AC','RJ','ES','MG','BA','SE','PE','AL','PB','CE','PI','MA','PA','AP','AM','RR','DF','GO','RO','TO','MT','MS','RS','PR','SC'],
                                         columns=['geolocation_state'])
    # Get the list of states that are not present in the current grouped data
    missing_states = all_geolocation_state[
        ~all_geolocation_state['geolocation_state'].isin(all_order_count_state[f'{seller_or_customer}_state'])]

    missing_states_df = pd.DataFrame({
        f'{seller_or_customer}_state': missing_states['geolocation_state'].unique(),
        'order_count': 0
    })
    bystate_df = pd.concat([all_order_count_state, missing_states_df], ignore_index=True)

    return bystate_df


def visualize_map_by_states(bystate_df, is_seller=True):
    # Clear any previous figure
    plt.clf()
    plt.close()

    seller_or_customer = 'seller' if is_seller else 'customer'

    # Load GeoJSON data into a GeoDataFrame
    geojson_url = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson'
    gdf = gpd.read_file(geojson_url)

    # Calculate quantiles for order_count
    quantiles = bystate_df['order_count'].quantile([0.25, 0.50, 0.75]).tolist()

    # # Find the index of value after the minimum value
    # asc_sorted = bystate_df.sort_values(by='order_count', ascending=True)
    # min_value = bystate_df['order_count'].min()
    # idx_last_min_val = asc_sorted.loc[bystate_df['order_count'] == min_value][
    #     f'{seller_or_customer}_state'].nunique()
    #
    # # Find the index of value before the maximum value
    # max_value = bystate_df['order_count'].max()
    # idx_last_max_val = (-1) - asc_sorted.loc[bystate_df['order_count'] == max_value][
    #     f'{seller_or_customer}_state'].nunique()
    #
    # # Create 1 bin for the min value, 1 bin for the max value, and 4 bins for the rest of quantile
    # bins = [min_value, asc_sorted.iloc[idx_last_min_val]['order_count'] - 0.1] + quantiles + [
    #     asc_sorted.iloc[idx_last_max_val]['order_count'], max_value] if (
    #             ~bystate_df['order_count'].max().isin(quantiles)) else [
    #     min_value - 0.1, max_value]

    # Sort the dataframe by 'order_count' in ascending order
    asc_sorted = bystate_df.sort_values(by='order_count', ascending=True)

    min_value = bystate_df['order_count'].min()
    # Find the index of value before the maximum value
    idx_last_min_val = asc_sorted.loc[bystate_df['order_count'] == min_value][
        f'{seller_or_customer}_state'].nunique()

    max_value = bystate_df['order_count'].max()
    # Find the index of value before the maximum value
    idx_last_max_val = (-1) - asc_sorted.loc[bystate_df['order_count'] == max_value][
        f'{seller_or_customer}_state'].nunique()

    # Start with the maximum number of bins: 6 bins (1 min, 4 quantiles, 1 max)
    num_quantiles = 3

    while num_quantiles >= 0:
        # Calculate quantiles dynamically based on the current number of quantiles
        quantiles = bystate_df['order_count'].quantile(
            [i / (num_quantiles + 1) for i in range(1, num_quantiles + 1)]
        ).tolist()

        # Create bins: 1 for min, 1 for max, and quantiles in between
        bins = ([min_value, asc_sorted.iloc[idx_last_min_val]['order_count'] - 0.1]
                + quantiles + [asc_sorted.iloc[idx_last_max_val]['order_count'], max_value])

        # Check if bins are strictly increasing (no duplicate values)
        if all(bins[i] < bins[i + 1] for i in range(len(bins) - 1)):
            break  # Exit the loop if bins are valid
        else:
            num_quantiles -= 1  # Reduce the number of quantiles

    # If no valid quantiles are found, fallback to 1 bin (min and max are the same)
    if num_quantiles == -1:
        bins = [min_value - 0.1, max_value]

    num_categories = len(bins) - 1

    # Assign bins for color mapping
    bystate_df['bin'] = pd.cut(
        bystate_df['order_count'],
        bins=bins,
        labels=False,
        include_lowest=True
    )

    # Merge GeoDataFrame for plotting
    gdf = gdf.merge(bystate_df, left_on='sigla', right_on=f'{seller_or_customer}_state')

    # Define a color map
    cmap = plt.cm.YlGn  # Matplotlib colormap
    colors = cmap(np.linspace(0, 1, num_categories))  # Generate color steps
    color_mapping = {i: colors[i] for i in range(num_categories)}  # Map bins to colors
    gdf['color'] = gdf['bin'].map(color_mapping)

    # Plot the GeoDataFrame
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', linewidth=0.5)

    # Add state names to the map
    for idx, row in gdf.iterrows():
        plt.annotate(text=row['sigla'], xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
                     horizontalalignment='center', fontsize=8, color='black')

    # Add a legend
    legend_labels = [f"{bins[i]:.1f} - {bins[i + 1]:.1f}" for i in range(num_categories)]
    for i, label in enumerate(legend_labels):
        ax.scatter([], [], color=colors[i], label=label)

    ax.legend(loc='lower left', title="Total Orders Count")
    ax.set_title(f"Number of {'Seller Sales' if is_seller else 'Customer Orders'} by Brazilian State")
    plt.axis("off")

    st.pyplot(plt)


def create_rating_summary_df(df):
    all_successful_orders = create_successful_orders_df(df)

    rating_summary_df = all_successful_orders.groupby('seller_id').agg(
        order_count=("order_id", "nunique"),
        average_rating=("review_score", "mean")
    ).reset_index()

    return rating_summary_df


def visualize_seller_rating(rating_summary_df):
    # Clear any previous figure
    plt.clf()
    plt.close()

    n_all_rating_below_2 = rating_summary_df.loc[
        (rating_summary_df['average_rating'] >= 1) & (rating_summary_df['average_rating'] < 2)]['seller_id'].nunique()

    n_all_rating_below_3 = rating_summary_df.loc[
        (rating_summary_df['average_rating'] >= 2) & (rating_summary_df['average_rating'] < 3)]['seller_id'].nunique()

    n_all_rating_below_4 = rating_summary_df.loc[
        (rating_summary_df['average_rating'] >= 3) & (rating_summary_df['average_rating'] < 4)]['seller_id'].nunique()

    n_all_rating_below_5 = rating_summary_df.loc[
        (rating_summary_df['average_rating'] >= 4) & (rating_summary_df['average_rating'] < 5)]['seller_id'].nunique()

    n_all_rating_perfect = rating_summary_df.loc[(rating_summary_df['average_rating'] == 5)]['seller_id'].nunique()

    labels = ['Rating 1 to <2', 'Rating 2 to <3', 'Rating 3 to <4', 'Rating 4 to <5', 'Rating 5']
    sizes = [n_all_rating_below_2,
             n_all_rating_below_3,
             n_all_rating_below_4,
             n_all_rating_below_5,
             n_all_rating_perfect]

    # Pie chart colormap
    cmap = plt.cm.RdYlGn
    colors = cmap(np.linspace(0, 1, len(sizes)))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.pie(sizes,
            labels=labels,
            colors=colors,
            autopct='%1.2f%%',
            shadow=False,
            startangle=140)

    plt.axis('equal')
    plt.title('Distribution of Seller Ratings')
    st.pyplot(fig)


def create_payment_preferences_df(df):
    all_successful_orders = create_successful_orders_df(df)

    # Group by payment sequential, so the payment_value is not duplicated
    all_successful_payment = all_successful_orders.groupby(['payment_type', 'order_id', 'payment_sequential']).agg({
        'payment_value': 'first'
    }).reset_index()

    payment_preferences_df = all_successful_payment.groupby('payment_type').agg(
        payment_count=('payment_type', 'count'),
        payment_value=('payment_value', 'sum')
    ).reset_index()

    # Calculate average value
    payment_preferences_df['average_value'] = round(payment_preferences_df['payment_value'] / payment_preferences_df['payment_count'], 2)

    # Ensure all payment types are included
    all_payment_types = ['credit_card', 'boleto', 'voucher', 'debit_card']
    default_df = pd.DataFrame({
        'payment_type': all_payment_types,
        'payment_count': 0,
        'payment_value': 0.0,
        'average_value': 0.0
    })

    # Merge the results with the default payment types
    payment_preferences_df = default_df.merge(
        payment_preferences_df, on='payment_type', how='left', suffixes=('_default', '_actual')
    )

    # Combine values (use 'actual' values if available, otherwise default values)
    payment_preferences_df['payment_count'] = payment_preferences_df['payment_count_actual'].fillna(0).astype(int)
    payment_preferences_df['payment_value'] = payment_preferences_df['payment_value_actual'].fillna(0.0)
    payment_preferences_df['average_value'] = payment_preferences_df['average_value_actual'].fillna(0.0)

    # Keep relevant columns
    payment_preferences_df = payment_preferences_df[['payment_type', 'payment_count', 'payment_value', 'average_value']]

    return payment_preferences_df


def visualize_payment_preferences(payment_preferences_df):
    # Clear any previous figure
    plt.clf()
    plt.close()

    n_all_boleto = (
                (payment_preferences_df.loc[payment_preferences_df['payment_type'] == 'boleto']['payment_count'].values[0] * 100) / (
            payment_preferences_df['payment_count'].sum()))

    n_all_cc = ((payment_preferences_df.loc[payment_preferences_df['payment_type'] == 'credit_card']['payment_count'].values[
                     0] * 100) / (payment_preferences_df['payment_count'].sum()))

    n_all_dc = ((payment_preferences_df.loc[payment_preferences_df['payment_type'] == 'debit_card']['payment_count'].values[
                     0] * 100) / (payment_preferences_df['payment_count'].sum()))

    n_all_voucher = (
                (payment_preferences_df.loc[payment_preferences_df['payment_type'] == 'voucher']['payment_count'].values[0] * 100) / (
            payment_preferences_df['payment_count'].sum()))

    labels = ['Boleto', 'Credit Card', 'Debit Card', 'Voucher']
    sizes = [n_all_boleto,
             n_all_cc,
             n_all_dc,
             n_all_voucher]

    # Pie chart colormap
    cmap = plt.cm.summer
    colors = cmap(np.linspace(0, 1, len(sizes)))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.pie(sizes,
            labels=labels,
            colors=colors,
            autopct='%1.2f%%',
            shadow=False,
            startangle=140)

    plt.axis('equal')
    plt.title('Customers Payment Type Preference')
    st.pyplot(fig)


def create_rfm_df(df):
    all_successful_orders = create_successful_orders_df(df)

    # Group by payment sequential, so the payment_value is not duplicated
    all_successful_payment = all_successful_orders.groupby(
        ['customer_unique_id', 'order_purchase_timestamp', 'order_id', 'payment_sequential']).agg({
        'payment_value': 'first'
    }).reset_index()

    rfm_df = all_successful_payment.groupby(by='customer_unique_id', as_index=False).agg(
        recent_order_date=('order_purchase_timestamp', 'max'),  # mengambil tanggal order terakhir
        frequency=('order_id', 'nunique'),  # menghitung jumlah order
        monetary=('payment_value', 'sum')  # menghitung jumlah revenue yang dihasilkan
    )

    # menghitung kapan terakhir pelanggan melakukan transaksi (hari)
    rfm_df['recent_order_date'] = rfm_df['recent_order_date'].dt.date
    recent_date = all_successful_orders['order_purchase_timestamp'].dt.date.max()
    rfm_df['recency'] = rfm_df['recent_order_date'].apply(lambda x: (recent_date - x).days)
    rfm_df.drop('recent_order_date', axis=1, inplace=True)

    return rfm_df


def visualize_rfm_analysis(rfm_df):
    # Clear any previous figure
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(35, 15))
    colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

    sns.barplot(y="recency", x="customer_unique_id", data=rfm_df.sort_values(by="recency", ascending=True).head(5),
                palette=colors, ax=ax[0], hue="customer_unique_id", legend=False)
    ax[0].set_ylabel(None)
    ax[0].set_xlabel("customer_unique_id", fontsize=30)
    ax[0].set_title("By Recency (days)", loc="center", fontsize=50)
    ax[0].tick_params(axis='y', labelsize=30)
    ax[0].tick_params(axis='x', labelsize=35, rotation=90)

    sns.barplot(y="frequency", x="customer_unique_id",
                data=rfm_df.sort_values(by="frequency", ascending=False).head(5), palette=colors, ax=ax[1],
                hue="customer_unique_id", legend=False)
    ax[1].set_ylabel(None)
    ax[1].set_xlabel("customer_unique_id", fontsize=30)
    ax[1].set_title("By Frequency", loc="center", fontsize=50)
    ax[1].tick_params(axis='y', labelsize=30)
    ax[1].tick_params(axis='x', labelsize=35, rotation=90)

    sns.barplot(y="monetary", x="customer_unique_id",
                data=rfm_df.sort_values(by="monetary", ascending=False).head(5), palette=colors, ax=ax[2],
                hue="customer_unique_id", legend=False)
    ax[2].set_ylabel(None)
    ax[2].set_xlabel("customer_unique_id")
    ax[2].set_title("By Monetary", loc="center", fontsize=50)
    ax[2].tick_params(axis='y', labelsize=30)
    ax[2].tick_params(axis='x', labelsize=35, rotation=90)

    st.pyplot(fig)


def create_customer_cluster_bins_df(df):
    new_rfm_df = create_rfm_df(df)

    new_rfm_df['avg_monetary'] = round(new_rfm_df['monetary'] / new_rfm_df['frequency'], 2)
    # Binning/manual clustering kolom frequency dan average monetary
    # Batas bawah dikurangi 1 (-1) untuk memastikan seluruh nilai termasuk di dalam cluster.
    # Statistik kolom diperiksa, jika nilai median sama dengan nilai maksimum, maka hanya bisa menggunakan 1 bin
    freq_bins = [new_rfm_df['frequency'].min() - 1, new_rfm_df['frequency'].median(),
                 new_rfm_df['frequency'].max()] if (
                new_rfm_df['frequency'].median() != new_rfm_df['frequency'].max()) else [
        new_rfm_df['frequency'].min() - 1, new_rfm_df['frequency'].max()]

    avg_mon_bins = [new_rfm_df['avg_monetary'].min() - 1, new_rfm_df['avg_monetary'].median(),
                    new_rfm_df['avg_monetary'].max()] if (
                new_rfm_df['avg_monetary'].median() != new_rfm_df['avg_monetary'].max()) else [
        new_rfm_df['avg_monetary'].min() - 1, new_rfm_df['avg_monetary'].max()]

    freq_labels = ['Low Frequency', 'High Frequency'] if (
                new_rfm_df['frequency'].median() != new_rfm_df['frequency'].max()) else ['Low Frequency']

    mon_labels = ['Low Monetary', 'High Monetary'] if (
                new_rfm_df['avg_monetary'].median() != new_rfm_df['avg_monetary'].max()) else ['Low Monetary']

    new_rfm_df['frequency_bin'] = pd.cut(new_rfm_df['frequency'], bins=freq_bins, labels=freq_labels)
    new_rfm_df['monetary_bin'] = pd.cut(new_rfm_df['avg_monetary'], bins=avg_mon_bins, labels=mon_labels)

    return new_rfm_df


def visualize_customer_clusters(new_rfm_df):
    # Clear any previous figure
    plt.clf()
    plt.close()

    # Create a new column for combined categories
    new_rfm_df['category'] = new_rfm_df['frequency_bin'].astype(str) + ' - ' + new_rfm_df['monetary_bin'].astype(str)
    category_counts = new_rfm_df['category'].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values,
                palette=sns.color_palette("pastel", n_colors=len(category_counts)), hue=category_counts.values)
    plt.title("Customer Segmentation by Frequency and Average Monetary", fontsize=16)
    plt.xlabel("Cluster Category", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)

    plt.tight_layout()
    st.pyplot(plt)


# Menyiapkan DataFrame
all_df = pd.read_csv(Path(__file__).parents[1] / "dashboard/all_data.csv")

datetime_columns = ['shipping_limit_date', 'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date', 'review_creation_date', 'review_answer_timestamp']
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column], errors='coerce')

zip_code_columns = ['customer_zip_code_prefix', 'seller_zip_code_prefix']
for column in zip_code_columns:
    all_df[column] = all_df[column].astype(int)
    all_df[column] = all_df[column].astype('object')

all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(inplace=True)


# Membuat Komponen Filter
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

with st.sidebar:
    # Menambahkan logo perusahaan
    st.image("https://github.com/tfajarama/python-data-analysis-dashboard/raw/main/final-project/olist-cover.png")

    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu', min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) &
                (all_df["order_purchase_timestamp"] <= str(end_date))]


# Menyiapkan dataframe yang dibutuhkan untuk visualisasi
daily_orders_df = create_daily_orders_df(main_df)
category_summary_df = create_category_summary_df(main_df)
delivery_summary_df = create_delivery_summary_df(main_df)
shipment_summary_df = create_shipment_summary_df(main_df)
seller_bystate_df = create_bystate_df(main_df, is_seller=True)
rating_summary_df = create_rating_summary_df(main_df)
customer_bystate_df = create_bystate_df(main_df, is_seller=False)
payment_preferences_df = create_payment_preferences_df(main_df)
rfm_df = create_rfm_df(main_df)
new_rfm_df = create_customer_cluster_bins_df(main_df)


# Melengkapi Dashboard dengan Berbagai Visualisasi Data
st.title('Olist Marketplace Dashboard :sparkles:')

# Pertanyaan Bisnis 1 & 2
st.header('Sales Performance')

# PB1: Bagaimana performa penjualan dan revenue perusahaan dalam beberapa waktu terakhir?
st.subheader('Daily Orders')

col1, col2 = st.columns(2)

with col1:
    total_successful_orders = daily_orders_df['successful_order_count'].sum()
    st.metric("Total Successful Orders", value=total_successful_orders)

with col2:
    total_revenue_gain = format_currency(daily_orders_df['revenue_gain'].sum(), "BRL", locale='es_CO')
    st.metric("Total Revenue Gain", value=total_revenue_gain)

col1, col2 = st.columns(2)

with col1:
    total_failed_orders = daily_orders_df['failed_order_count'].sum()
    st.metric("Total Failed Orders", value=total_failed_orders)

with col2:
    total_revenue_loss = format_currency(daily_orders_df['revenue_loss'].sum(), "BRL", locale='es_CO')
    st.metric("Total Revenue Loss", value=f"-{total_revenue_loss}")

visualize_daily_orders(daily_orders_df)


# PB2: Kategori produk apa yang paling banyak dan paling sedikit penjualan serta revenuenya?
st.subheader("Best & Worst Performing Product by Number of Sales")
visualize_product_performance(category_summary_df)


# Pertanyaan Bisnis 3 & 4
st.header("Logistics Performance")

# PB3: Bagaimana ketepatan waktu delivery kurir dengan waktu estimasinya?
st.subheader("Carrier Delivery Performance")
visualize_delivery_performance(delivery_summary_df)

# PB4: Bagaimana ketepatan waktu shipping penjual dengan batas waktunya?
st.subheader("Seller Shipment Performance")
visualize_shipment_performance(shipment_summary_df)


# Pertanyaan Bisnis 5 & 6
st.header("Seller Analytics")

# PB5: Bagaimana persebaran lokasi penjual dan hasil penjualannya?
st.subheader("Seller Location Distribution")
visualize_map_by_states(seller_bystate_df, is_seller=True)

# PB6: Bagaimana persebaran rating penjual?
st.subheader("Seller Rating Distribution")
visualize_seller_rating(rating_summary_df)

# Pertanyaan Bisnis (PB) 7-10
st.header("Customer Analytics")

# PB7: Bagaimana persebaran lokasi pelanggan dan total pembeliannya?
st.subheader("Customer Location Distribution")
visualize_map_by_states(customer_bystate_df, is_seller=False)

# PB8: Bagaimana preferensi pelanggan dalam melakukan pembayaran transaksi?
st.subheader("Customer Payment Preferences")
visualize_payment_preferences(payment_preferences_df)

# PB9: Kapan terakhir pelanggan melakukan transaksi?
#       Seberapa sering seorang pelanggan melakukan pembelian dalam beberapa bulan terakhir?
#       Berapa banyak uang yang dihabiskan pelanggan dalam beberapa bulan terakhir?
st.subheader("Best Customer Based on RFM Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    avg_recency = round(rfm_df.recency.mean(), 1)
    st.metric("Average Recency (days)", value=avg_recency)

with col2:
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    st.metric("Average Frequency", value=avg_frequency)

with col3:
    avg_frequency = format_currency(rfm_df.monetary.mean(), "AUD", locale='es_CO')
    st.metric("Average Monetary", value=avg_frequency)

visualize_rfm_analysis(rfm_df)


# PB10: Bagaimana persebaran kelompok pelanggan berdasarkan frekuensi pembelian dan nominal transaksi?
st.subheader("Customer Segmentation by Frequency and Average Monetary")
visualize_customer_clusters(new_rfm_df)


st.caption('Copyright (c) @tfajarama Dicoding Data Analysis Final Project Submission 2024')



# st.subheader("Customer Demographics")
#
# col1, col2 = st.columns(2)
#
# with col1:
#     fig, ax = plt.subplots(figsize=(20, 10))
#
#     sns.barplot(
#         y="customer_count",
#         x="gender",
#         data=bygender_df.sort_values(by="customer_count", ascending=False),
#         palette=colors[0:3],
#         ax=ax,
#         hue="gender",
#         legend=False
#     )
#     ax.set_title("Number of Customer by Gender", loc="center", fontsize=50)
#     ax.set_ylabel(None)
#     ax.set_xlabel(None)
#     ax.tick_params(axis='x', labelsize=35)
#     ax.tick_params(axis='y', labelsize=30)
#     st.pyplot(fig)
#
# with col2:
#     fig, ax = plt.subplots(figsize=(20, 10))
#
#     colors = ["#D3D3D3", "#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
#
#     sns.barplot(
#         y="customer_count",
#         x="age_group",
#         data=byage_df.sort_values(by="age_group", ascending=False),
#         palette=colors[0:3],
#         ax=ax,
#         hue="age_group",
#         legend=False
#     )
#     ax.set_title("Number of Customer by Age", loc="center", fontsize=50)
#     ax.set_ylabel(None)
#     ax.set_xlabel(None)
#     ax.tick_params(axis='x', labelsize=35)
#     ax.tick_params(axis='y', labelsize=30)
#     st.pyplot(fig)
#
# fig, ax = plt.subplots(figsize=(20, 10))
# colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
# sns.barplot(
#     x="customer_count",
#     y="state",
#     data=bystate_df.sort_values(by="customer_count", ascending=False),
#     palette=colors,
#     ax=ax,
#     hue="state",
#     legend=False
# )
# ax.set_title("Number of Customer by States", loc="center", fontsize=30)
# ax.set_ylabel(None)
# ax.set_xlabel(None)
# ax.tick_params(axis='y', labelsize=20)
# ax.tick_params(axis='x', labelsize=15)
# st.pyplot(fig)