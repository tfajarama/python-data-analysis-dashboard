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
    Filter the DataFrame to include only successful orders.

    Parameters:
    df (DataFrame): The input DataFrame containing order data with a column 'order_status'.

    Returns:
    DataFrame: A DataFrame containing only the orders that are not 'canceled' or 'unavailable'.
    """
    return df[~df['order_status'].isin(['canceled', 'unavailable'])]


def create_failed_orders_df(df):
    """
    Filter the DataFrame to include only failed orders.

    Parameters:
    df (DataFrame): The input DataFrame containing order data with a column 'order_status'.

    Returns:
    DataFrame: A DataFrame containing only the orders that are 'canceled' or 'unavailable'.
    """
    return df[df['order_status'].isin(['canceled', 'unavailable'])]


def create_daily_orders_df(df):
    """
    Create a DataFrame with daily aggregated order data, including total orders, successful orders,
    failed orders, revenue gain, and revenue loss.

    Parameters:
    df (DataFrame): The input DataFrame containing order data with columns 'order_id',
                    'order_purchase_timestamp', 'payment_sequential', and 'payment_value'.

    Returns:
    DataFrame: A DataFrame with daily aggregated data, including:
               - 'order_purchase_timestamp': The date of the orders.
               - 'order_count': The total number of orders per day.
               - 'successful_order_count': The number of successful orders per day.
               - 'revenue_gain': The total revenue from successful orders per day.
               - 'failed_order_count': The number of failed orders per day.
               - 'revenue_loss': The total revenue loss from failed orders per day.
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
    Visualize the daily count of successful orders using a line plot.

    Parameters:
    daily_orders_df (DataFrame): The input DataFrame containing daily aggregated order data with
                                 columns 'order_purchase_timestamp' and 'successful_order_count'.

    Returns:
    None: This function does not return any value. It displays a plot of the daily successful order count.
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
    plt.xlabel('Order Purchase Date')
    plt.ylabel('Number of Orders')

    st.pyplot(fig)


def create_category_summary_df(df):
    """
    Create a summary DataFrame with total orders and total revenue for each product category.

    Parameters:
    df (DataFrame): The input DataFrame containing order data with columns 'product_category_name_english',
                    'order_id', 'payment_sequential', and 'payment_value'.

    Returns:
    DataFrame: A DataFrame with aggregated data for each product category, including:
               - 'product_category_name_english': The name of the product category.
               - 'total_orders': The total number of unique orders for each product category.
               - 'total_revenue': The total revenue for each product category.
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


def visualize_product_performance(category_summary_df, is_orders=True):
    """
    Visualize the performance of product categories using bar plots for the best and worst performing products.

    Parameters:
    category_summary_df (DataFrame): The input DataFrame containing summary data for product categories with
                                     columns 'product_category_name_english', 'total_orders', and 'total_revenue'.
    is_orders (bool): If True, visualize the number of orders. If False, visualize the revenue. Default is True.

    Returns:
    None: This function does not return any value. It displays bar plots for the best and worst performing products.
    """
    # Clear any previous figure
    plt.clf()
    plt.close()

    orders_or_revenue = 'orders' if is_orders else 'revenue'

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))

    colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

    # Best Performing Product
    sns.barplot(x=f"total_{orders_or_revenue}", y="product_category_name_english",
                data=category_summary_df.sort_values(by=f'total_{orders_or_revenue}', ascending=False).head(5),
                palette=colors, ax=ax[0], hue="product_category_name_english", legend=False)
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(f"{'Number of Sales' if is_orders else 'Revenue Gain Nominal (in multiplies of BRL1.000.000,00)'}", fontsize=30)
    ax[0].set_title("Best Performing Product", loc="center", fontsize=50)
    ax[0].tick_params(axis='y', labelsize=35)
    ax[0].tick_params(axis='x', labelsize=30)

    # Worst Performing Product
    sns.barplot(x=f"total_{orders_or_revenue}", y="product_category_name_english",
                data=category_summary_df.sort_values(by=f'total_{orders_or_revenue}', ascending=True).head(5),
                palette=colors, ax=ax[1], hue="product_category_name_english", legend=False)
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(f"{'Number of Sales' if is_orders else 'Revenue Gain Nominal (in BRL)'}", fontsize=30)
    ax[1].invert_xaxis()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_title("Worst Performing Product", loc="center", fontsize=50)
    ax[1].tick_params(axis='y', labelsize=35)
    ax[1].tick_params(axis='x', labelsize=30)

    st.pyplot(fig)


def create_delivery_summary_df(df):
    """
    Create a summary DataFrame with delivery details, including actual delivery dates, estimated delivery dates,
    and the difference between them.

    Parameters:
    df (DataFrame): The input DataFrame containing order data with columns 'order_id',
                    'order_delivered_customer_date', and 'order_estimated_delivery_date'.

    Returns:
    DataFrame: A DataFrame with delivery summary data, including:
               - 'order_id': The unique identifier for each order.
               - 'delivered_date': The actual delivery date of the order.
               - 'estimated_date': The estimated delivery date of the order.
               - 'delivery_estimation_diff': The difference in days between the actual and estimated delivery dates.
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
    Visualize the performance of delivery times using a histogram and a normal distribution curve.

    Parameters:
    delivery_summary_df (DataFrame): The input DataFrame containing delivery summary data with
                                     a column 'delivery_estimation_diff' representing the difference
                                     in days between the actual delivery date and the estimated delivery date.

    Returns:
    None: This function does not return any value. It displays a histogram of the delivery estimation differences.
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
    """
    Create a summary DataFrame with shipment details, including actual shipment dates, shipping limit dates,
    and the difference between them.

    Parameters:
    df (DataFrame): The input DataFrame containing order data with columns 'order_id',
                    'order_delivered_carrier_date', and 'shipping_limit_date'.

    Returns:
    DataFrame: A DataFrame with shipment summary data, including:
               - 'order_id': The unique identifier for each order.
               - 'shipped_date': The actual shipment date of the order.
               - 'limit_date': The shipping limit date of the order.
               - 'shipment_limit_diff': The difference in days between the actual shipment date and the shipping limit date.
    """
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
    """
    Visualize the performance of shipment times using a histogram and a normal distribution curve.

    Parameters:
    shipment_summary_df (DataFrame): The input DataFrame containing shipment summary data with
                                     a column 'shipment_limit_diff' representing the difference
                                     in days between the actual shipment date and the shipping limit date.

    Returns:
    None: This function does not return any value. It displays a histogram of the shipment limit differences.
    """
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
    """
    Create a DataFrame with the count of orders by state, including states with zero orders.

    Parameters:
    df (DataFrame): The input DataFrame containing order data with columns 'order_id' and either
                    'seller_state' or 'customer_state'.
    is_seller (bool): If True, aggregate orders by seller state. If False, aggregate orders by customer state.
                      Default is True.

    Returns:
    DataFrame: A DataFrame with the count of orders by state, including states with zero orders, with columns:
               - 'seller_state' or 'customer_state': The state of the seller or customer.
               - 'order_count': The total number of unique orders for each state.
    """
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
    """
    Visualize the number of orders by Brazilian state using a choropleth map.

    Parameters:
    bystate_df (DataFrame): The input DataFrame containing order counts by state with columns
                            'seller_state' or 'customer_state' and 'order_count'.
    is_seller (bool): If True, visualize seller sales by state. If False, visualize customer orders by state.
                      Default is True.

    Returns:
    None: This function does not return any value. It displays a choropleth map of the order counts by state.
    """
    # Clear any previous figure
    plt.clf()
    plt.close()

    seller_or_customer = 'seller' if is_seller else 'customer'

    # Load GeoJSON data into a GeoDataFrame
    geojson_url = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson'
    gdf = gpd.read_file(geojson_url)

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
    """
    Create a summary DataFrame with the count of orders and average rating for each seller.

    Parameters:
    df (DataFrame): The input DataFrame containing order data with columns 'seller_id', 'order_id',
                    and 'review_score'.

    Returns:
    DataFrame: A DataFrame with rating summary data for each seller, including:
               - 'seller_id': The unique identifier for each seller.
               - 'order_count': The total number of unique orders for each seller.
               - 'average_rating': The average review score for each seller.
    """
    all_successful_orders = create_successful_orders_df(df)

    rating_summary_df = all_successful_orders.groupby('seller_id').agg(
        order_count=("order_id", "nunique"),
        average_rating=("review_score", "mean")
    ).reset_index()

    return rating_summary_df


def visualize_seller_rating(rating_summary_df):
    """
    Visualize the distribution of seller ratings using a pie chart.

    Parameters:
    rating_summary_df (DataFrame): The input DataFrame containing rating summary data for sellers with
                                   columns 'seller_id' and 'average_rating'.

    Returns:
    None: This function does not return any value. It displays a pie chart of the distribution of seller ratings.
    """
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

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.pie(sizes,
            labels=labels,
            colors=colors,
            autopct='%1.2f%%',
            shadow=False,
            startangle=140)

    plt.axis('equal')
    st.pyplot(fig)


def create_payment_preferences_df(df):
    """
    Create a summary DataFrame with payment preferences, including the count, total value, and average value
    of payments for each payment type.

    Parameters:
    df (DataFrame): The input DataFrame containing order data with columns 'payment_type', 'order_id',
                    'payment_sequential', and 'payment_value'.

    Returns:
    DataFrame: A DataFrame with payment preferences data for each payment type, including:
               - 'payment_type': The type of payment (e.g., credit_card, boleto, voucher, debit_card).
               - 'payment_count': The total number of payments for each payment type.
               - 'payment_value': The total value of payments for each payment type.
               - 'average_value': The average value of payments for each payment type.
    """
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
    """
    Visualize the distribution of payment preferences using a pie chart.

    Parameters:
    payment_preferences_df (DataFrame): The input DataFrame containing payment preferences data with
                                        columns 'payment_type' and 'payment_count'.

    Returns:
    None: This function does not return any value. It displays a pie chart of the distribution of payment preferences.
    """
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

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.pie(sizes,
            labels=labels,
            colors=colors,
            autopct='%1.2f%%',
            shadow=False,
            startangle=140)

    plt.axis('equal')
    st.pyplot(fig)


def create_rfm_df(df):
    """
    Create a DataFrame with RFM (Recency, Frequency, Monetary) metrics for each customer.

    Parameters:
    df (DataFrame): The input DataFrame containing order data with columns 'customer_unique_id',
                    'order_purchase_timestamp', 'order_id', 'payment_sequential', and 'payment_value'.

    Returns:
    DataFrame: A DataFrame with RFM metrics for each customer, including:
               - 'customer_unique_id': The unique identifier for each customer.
               - 'recency': The number of days since the customer's most recent order.
               - 'frequency': The total number of unique orders placed by the customer.
               - 'monetary': The total revenue generated by the customer.
    """
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
    """
    Visualize the RFM (Recency, Frequency, Monetary) analysis using bar plots.

    Parameters:
    rfm_df (DataFrame): The input DataFrame containing RFM metrics for customers with columns
                        'customer_unique_id', 'recency', 'frequency', and 'monetary'.

    Returns:
    None: This function does not return any value. It displays bar plots for the top 5 customers by recency,
          frequency, and monetary value.
    """
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
    """
    Create a DataFrame with customer clusters based on frequency and average monetary value.

    Parameters:
    df (DataFrame): The input DataFrame containing order data with columns 'customer_unique_id',
                    'order_purchase_timestamp', 'order_id', 'payment_sequential', and 'payment_value'.

    Returns:
    DataFrame: A DataFrame with customer clusters, including:
               - 'customer_unique_id': The unique identifier for each customer.
               - 'frequency': The total number of unique orders placed by the customer.
               - 'monetary': The total revenue generated by the customer.
               - 'avg_monetary': The average revenue per order for the customer.
               - 'frequency_bin': The cluster label for frequency (e.g., 'Low Frequency', 'High Frequency').
               - 'monetary_bin': The cluster label for average monetary value (e.g., 'Low Monetary', 'High Monetary').
    """
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
    """
    Visualize customer clusters based on frequency and monetary value using a bar plot.

    Parameters:
    new_rfm_df (DataFrame): The input DataFrame containing customer clusters with columns
                            'frequency_bin', 'monetary_bin', and 'category'.

    Returns:
    None: This function does not return any value. It displays a bar plot of the customer clusters.
    """
    # Clear any previous figure
    plt.clf()
    plt.close()

    # Create a new column for combined categories
    new_rfm_df['category'] = new_rfm_df['frequency_bin'].astype(str) + ' - ' + new_rfm_df['monetary_bin'].astype(str)
    category_counts = new_rfm_df['category'].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values,
                palette=sns.color_palette("pastel", n_colors=len(category_counts)), hue=category_counts.values)
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

    # Navigation links
    st.markdown("""
        <style>
        .nav-link {
            display: block;
            padding: 8px;
            text-decoration: none;
            color: inherit; /* Inherit text color based on theme */
        }
        .nav-link:hover {
            background-color: var(--primary-background-color); /* Use theme's primary background color */
        }
        .sticky-nav {
            position: -webkit-sticky; /* For Safari */
            position: sticky;
            top: 0;
            background-color: var(--secondary-background-color); /* Use theme's secondary background color */
            padding: 10px;
            z-index: 1000;
        }
        </style>
        <div class="sticky-nav">
            <a class="nav-link" href="#section1">Sales Performance</a>
            <a class="nav-link" href="#section2">Logistics Performance</a>
            <a class="nav-link" href="#section3">Seller Analytics</a>
            <a class="nav-link" href="#section4">Customer Analytics</a>
        </div>
    """, unsafe_allow_html=True)


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
st.markdown('<div id="section1"></div>', unsafe_allow_html=True)
st.header('Sales Performance')

# PB1: Bagaimana performa penjualan dan revenue perusahaan dalam beberapa waktu terakhir?
st.subheader('Daily Orders')

col1, col2 = st.columns(2)

with col1:
    total_successful_orders = daily_orders_df['successful_order_count'].sum()
    successful_orders_percentage = round(((total_successful_orders * 100) / daily_orders_df['order_count'].sum()), 2)
    st.metric("Total Successful Orders", value=f"{total_successful_orders} ({successful_orders_percentage}%)")

with col2:
    total_revenue_gain = format_currency(daily_orders_df['revenue_gain'].sum(), "BRL", locale='es_CO')
    st.metric("Total Revenue Gain", value=total_revenue_gain)

col1, col2 = st.columns(2)

with col1:
    total_failed_orders = daily_orders_df['failed_order_count'].sum()
    failed_orders_percentage = round(((total_failed_orders * 100) / daily_orders_df['order_count'].sum()), 2)
    st.metric("Total Failed Orders", value=f"{total_failed_orders} ({failed_orders_percentage}%)")

with col2:
    total_revenue_loss = format_currency(daily_orders_df['revenue_loss'].sum(), "BRL", locale='es_CO')
    st.metric("Total Revenue Loss", value=f"-{total_revenue_loss}")

visualize_daily_orders(daily_orders_df)


# PB2: Kategori produk apa yang paling banyak dan paling sedikit penjualan serta revenuenya?
st.subheader("Best & Worst Performing Product by Number of Sales")

col1, col2 = st.columns(2)

with col1:
    best_sales_product = \
    category_summary_df.sort_values(by='total_orders', ascending=False).head(1)['total_orders'].values[0]
    st.metric("Best Performing Sales", value=f"{best_sales_product} orders")

with col2:
    worst_sales_product = \
    category_summary_df.sort_values(by='total_orders', ascending=True).head(1)['total_orders'].values[0]
    st.metric("Worst Performing Sales", value=f"{worst_sales_product} orders")

visualize_product_performance(category_summary_df, is_orders=True)

st.subheader("Best & Worst Performing Product by Revenue Gain Nominal")

col1, col2 = st.columns(2)

with col1:
    best_revenue_product = \
    category_summary_df.sort_values(by='total_revenue', ascending=False).head(1)['total_revenue'].values[0]
    st.metric("Best Performing Revenue", value=format_currency(best_revenue_product, "BRL", locale='es_CO'))

with col2:
    worst_revenue_product = \
    category_summary_df.sort_values(by='total_revenue', ascending=True).head(1)['total_revenue'].values[0]
    st.metric("Worst Performing Revenue", value=format_currency(worst_revenue_product, "BRL", locale='es_CO'))

visualize_product_performance(category_summary_df, is_orders=False)


# Pertanyaan Bisnis 3 & 4
st.markdown('<div id="section2"></div>', unsafe_allow_html=True)
st.header("Logistics Performance")

# PB3: Bagaimana ketepatan waktu delivery kurir dengan waktu estimasinya?
st.subheader("Carrier Delivery Performance")

col1, col2, col3 = st.columns(3)

with col1:
    deliv_before_time = delivery_summary_df.loc[delivery_summary_df['delivery_estimation_diff'] < 0]['order_id'].nunique()
    deliv_before_time_percentage = round(((deliv_before_time * 100) / delivery_summary_df.shape[0]), 2)
    st.metric("Deliveries Before Estimated Date", value=f"{deliv_before_time} ({deliv_before_time_percentage}%)")

with col2:
    deliv_on_time = delivery_summary_df.loc[delivery_summary_df['delivery_estimation_diff'] == 0]['order_id'].nunique()
    deliv_on_time_percentage = round(((deliv_on_time * 100) / delivery_summary_df.shape[0]), 2)
    st.metric("Deliveries On Estimated Date", value=f"{deliv_on_time} ({deliv_on_time_percentage}%)")

with col3:
    deliv_after_time = delivery_summary_df.loc[delivery_summary_df['delivery_estimation_diff'] > 0]['order_id'].nunique()
    deliv_after_time_percentage = round(((deliv_after_time * 100) / delivery_summary_df.shape[0]), 2)
    st.metric("Deliveries After Estimated Date", value=f"{deliv_after_time} ({deliv_after_time_percentage}%)")

col1, col2 = st.columns(2)

with col1:
    delivery_min_val = delivery_summary_df['delivery_estimation_diff'].min()
    st.metric("Minimum Delivery Date (Before Estimated Date)", value=f"{delivery_min_val} days")

with  col2:
    delivery_max_val = delivery_summary_df['delivery_estimation_diff'].max()
    st.metric("Maximum Delivery Date (After Estimated Date)", value=f"{delivery_max_val} days")

visualize_delivery_performance(delivery_summary_df)


# PB4: Bagaimana ketepatan waktu shipping penjual dengan batas waktunya?
st.subheader("Seller Shipment Performance")

col1, col2, col3 = st.columns(3)

with col1:
    shipment_before_time = shipment_summary_df.loc[shipment_summary_df['shipment_limit_diff'] < 0]['order_id'].nunique()
    shipment_before_time_percentage = round(((shipment_before_time * 100) / shipment_summary_df.shape[0]), 2)
    st.metric("Shipment Before Limit Date", value=f"{shipment_before_time} ({shipment_before_time_percentage}%)")

with col2:
    shipment_on_time = shipment_summary_df.loc[shipment_summary_df['shipment_limit_diff'] == 0]['order_id'].nunique()
    shipment_on_time_percentage = round(((shipment_on_time * 100) / shipment_summary_df.shape[0]), 2)
    st.metric("Shipment On Limit Date", value=f"{shipment_on_time} ({shipment_on_time_percentage}%)")

with col3:
    shipment_after_time = shipment_summary_df.loc[shipment_summary_df['shipment_limit_diff'] > 0]['order_id'].nunique()
    shipment_after_time_percentage = round(((shipment_after_time * 100) / shipment_summary_df.shape[0]), 2)
    st.metric("Shipment After Limit Date", value=f"{shipment_after_time} ({shipment_after_time_percentage}%)")

col1, col2 = st.columns(2)

with col1:
    shipment_min_val = shipment_summary_df['shipment_limit_diff'].min()
    st.metric("Minimum Shipment Date (Before Limit Date)", value=f"{shipment_min_val} days")

with  col2:
    shipment_max_val = shipment_summary_df['shipment_limit_diff'].max()
    st.metric("Maximum Shipment Date (After Limit Date)", value=f"{shipment_max_val} days")

visualize_shipment_performance(shipment_summary_df)


# Pertanyaan Bisnis 5 & 6
st.markdown('<div id="section3"></div>', unsafe_allow_html=True)
st.header("Seller Analytics")

# PB5: Bagaimana persebaran lokasi penjual dan hasil penjualannya?
st.subheader("Seller Location Distribution")

col1, col2 = st.columns(2)

with col1:
    seller_most_state = seller_bystate_df.sort_values(by='order_count', ascending=False).head(1)['seller_state'].values[0]
    n_state_most_seller = seller_bystate_df.loc[seller_bystate_df['order_count'] == seller_bystate_df['order_count'].max()]['seller_state'].nunique()
    n_others_most = f" and {n_state_most_seller - 1} others" if n_state_most_seller > 1 else ''
    st.metric("Seller State With Most Orders", value=f"State {seller_most_state} {n_others_most}")

with col2:
    seller_least_state = seller_bystate_df.sort_values(by='order_count', ascending=True).head(1)['seller_state'].values[0]
    n_state_least_seller = seller_bystate_df.loc[seller_bystate_df['order_count'] == seller_bystate_df['order_count'].min()]['seller_state'].nunique()
    n_others_least = f" and {n_state_least_seller - 1} others" if n_state_least_seller > 1 else ''
    st.metric("Seller State With Least Orders", value=f"State {seller_least_state}{n_others_least}")

visualize_map_by_states(seller_bystate_df, is_seller=True)


# PB6: Bagaimana persebaran rating penjual?
st.subheader("Seller Rating Distribution")

avg_rating = round(rating_summary_df['average_rating'].mean(), 2)
st.metric("Average Seller Rating", value=f"{avg_rating} ({rating_summary_df.shape[0]} sellers)")

visualize_seller_rating(rating_summary_df)


# Pertanyaan Bisnis (PB) 7-10
st.markdown('<div id="section4"></div>', unsafe_allow_html=True)
st.header("Customer Analytics")

# PB7: Bagaimana persebaran lokasi pelanggan dan total pembeliannya?
st.subheader("Customer Location Distribution")

col1, col2 = st.columns(2)

with col1:
    cust_most_state = customer_bystate_df.sort_values(by='order_count', ascending=False).head(1)['customer_state'].values[0]
    n_state_most_cust = customer_bystate_df.loc[customer_bystate_df['order_count'] == customer_bystate_df['order_count'].max()]['customer_state'].nunique()
    n_others_most = f" and {n_state_most_cust - 1} others" if n_state_most_cust > 1 else ''
    st.metric("Customer State With Most Orders", value=f"State {cust_most_state} {n_others_most}")

with col2:
    cust_least_state = customer_bystate_df.sort_values(by='order_count', ascending=True).head(1)['customer_state'].values[0]
    n_state_least_cust = customer_bystate_df.loc[customer_bystate_df['order_count'] == customer_bystate_df['order_count'].min()]['customer_state'].nunique()
    n_others_least = f" and {n_state_least_cust - 1} others" if n_state_least_cust > 1 else ''
    st.metric("Customer State With Least Orders", value=f"State {cust_least_state}{n_others_least}")

visualize_map_by_states(customer_bystate_df, is_seller=False)


# PB8: Bagaimana preferensi pelanggan dalam melakukan pembayaran transaksi?
st.subheader("Customer Payment Preferences")

col1, col2 = st.columns(2)

with col1:
    avg_cc = payment_preferences_df.loc[payment_preferences_df['payment_type'] == 'credit_card']['average_value'].values[0]
    st.metric("Credit Card Average Transaction", value=format_currency(avg_cc, "BRL", locale='es_CO'))

with col2:
    avg_dc = payment_preferences_df.loc[payment_preferences_df['payment_type'] == 'debit_card']['average_value'].values[0]
    st.metric("Debit Card Average Transaction", value=format_currency(avg_dc, "BRL", locale='es_CO'))

col1, col2 = st.columns(2)

with col1:
    avg_boleto = payment_preferences_df.loc[payment_preferences_df['payment_type'] == 'boleto']['average_value'].values[0]
    st.metric("Boleto Average Transaction", value=format_currency(avg_boleto, "BRL", locale='es_CO'))

with col2:
    avg_voucher = payment_preferences_df.loc[payment_preferences_df['payment_type'] == 'voucher']['average_value'].values[0]
    st.metric("Voucher Average Transaction", value=format_currency(avg_voucher, "BRL", locale='es_CO'))

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
    avg_frequency = format_currency(rfm_df.monetary.mean(), "BRL", locale='es_CO')
    st.metric("Average Monetary", value=avg_frequency)

visualize_rfm_analysis(rfm_df)


# PB10: Bagaimana persebaran kelompok pelanggan berdasarkan frekuensi pembelian dan nominal transaksi?
st.subheader("Customer Segmentation by Frequency and Average Monetary")

col1, col2 = st.columns(2)

with col1:
    min_bin = new_rfm_df.loc[new_rfm_df['frequency_bin'] == 'Low Frequency']['frequency'].min()
    max_bin = new_rfm_df.loc[new_rfm_df['frequency_bin'] == 'Low Frequency']['frequency'].max()
    low_f_metric = f"{min_bin}—{max_bin}" if min_bin is not None and max_bin is not None else "Not Available"
    st.metric("Low Frequency Range", value=low_f_metric)

with col2:
    min_bin = new_rfm_df.loc[new_rfm_df['frequency_bin'] == 'High Frequency']['frequency'].min()
    max_bin = new_rfm_df.loc[new_rfm_df['frequency_bin'] == 'High Frequency']['frequency'].max()
    high_f_metric = f"{min_bin}—{max_bin}" if min_bin is not None and max_bin is not None else "Not Available"
    st.metric("High Frequency Range", value=high_f_metric)

col1, col2 = st.columns(2)

with col1:
    min_bin = new_rfm_df.loc[new_rfm_df['monetary_bin'] == 'Low Monetary']['avg_monetary'].min()
    max_bin = new_rfm_df.loc[new_rfm_df['monetary_bin'] == 'Low Monetary']['avg_monetary'].max()
    low_f_metric = f"{min_bin}—{max_bin}" if min_bin is not None and max_bin is not None else "Not Available"
    st.metric("Low Monetary Range (in multiplies of BRL1,00)", value=low_f_metric)

with col2:
    min_bin = new_rfm_df.loc[new_rfm_df['monetary_bin'] == 'High Monetary']['avg_monetary'].min()
    max_bin = new_rfm_df.loc[new_rfm_df['monetary_bin'] == 'High Monetary']['avg_monetary'].max()
    high_f_metric = f"{min_bin}—{max_bin}" if min_bin is not None and max_bin is not None else "Not Available"
    st.metric("High Monetary Range (in multiplies of BRL1,00)", value=high_f_metric)

visualize_customer_clusters(new_rfm_df)


st.caption(f'Copyright (c) [@tfajarama](https://github.com/tfajarama/python-data-analysis-dashboard) Final Project Submission [Dicoding Data Analysis](https://www.dicoding.com/academies/555-belajar-analisis-data-dengan-python) 2024')