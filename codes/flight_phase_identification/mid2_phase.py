"""
Aircraft Flight Phase Classification and Visualization

This script queries BigQuery for flight data, classifies flight phases,
and visualizes altitude profiles for randomly selected flights.
"""

import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fire
from google.cloud import bigquery
from matplotlib.ticker import FuncFormatter


def format_altitude(x, pos):
    """Format altitude values with k/M suffixes for better readability."""
    if x >= 1000000:
        return f'{x/1000000:.1f}M'
    elif x >= 1000:
        return f'{x/1000:.0f}k'
    else:
        return f'{x:.0f}'


def verify_credentials(credentials_path):
    """
    Verify that the credentials file exists and set environment variable.
    
    Args:
        credentials_path: Path to the service account JSON key file
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    if not os.path.exists(credentials_path):
        print(f"❌ Error: Credentials file not found at: {credentials_path}")
        return False
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    print(f"✅ Credentials verified: {credentials_path}")
    return True


def create_bigquery_client(project_id):
    """
    Create and return a BigQuery client.
    
    Args:
        project_id: Google Cloud project ID
        
    Returns:
        bigquery.Client: Initialized BigQuery client
    """
    try:
        client = bigquery.Client(project=project_id)
        print(f"✅ BigQuery client created for project: {project_id}")
        return client
    except Exception as e:
        print(f"❌ Error creating BigQuery client: {e}")
        raise


def query_and_classify_flights(client, limit=None):
    """
    Query flight data from BigQuery and classify flight phases.
    
    Args:
        client: BigQuery client instance
        limit: Optional limit on number of rows to return (None for all data)
        
    Returns:
        pandas.DataFrame: Flight data with classified phases
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    query = f"""
    WITH aggregated AS (
      SELECT
        HexIdent,
        FlightID,
        Date_MSG_Generated,
        Time_MSG_Generated,
        Time_MSG_Logged,
        MAX(Callsign) AS Callsign,
        MAX(Altitude) AS Altitude,
        MAX(GroundSpeed) AS GroundSpeed,
        MAX(Latitude) AS Latitude,
        MAX(Longitude) AS Longitude,
        TIMESTAMP(DATETIME(Date_MSG_Generated, Time_MSG_Generated)) AS Generated_TS
      FROM `iitp-class-team-2-473114.SBS_Data.FirstRun`
      GROUP BY
        HexIdent,
        FlightID,
        Date_MSG_Generated,
        Time_MSG_Generated,
        Time_MSG_Logged
    ),

    with_duration AS (
      SELECT
        *,
        TIMESTAMP_DIFF(
          MAX(Generated_TS) OVER (PARTITION BY Callsign),
          MIN(Generated_TS) OVER (PARTITION BY Callsign),
          SECOND
        ) AS flight_duration_sec
      FROM aggregated
      WHERE
        Callsign IS NOT NULL
        AND Callsign NOT IN ('0', '0000', '00000', '0000000', '00000000')
    ),

    with_duration_filtered AS (
      SELECT *
      FROM with_duration
      WHERE flight_duration_sec >= 1800
    ),

    with_rates AS (
      SELECT
        *,
        TIMESTAMP_DIFF(
          Generated_TS,
          LAG(Generated_TS) OVER (PARTITION BY Callsign ORDER BY Generated_TS),
          SECOND
        ) AS delta_t_sec,

        SAFE_DIVIDE(
          Altitude - LAG(Altitude) OVER (PARTITION BY Callsign ORDER BY Generated_TS),
          TIMESTAMP_DIFF(
            Generated_TS,
            LAG(Generated_TS) OVER (PARTITION BY Callsign ORDER BY Generated_TS),
            SECOND
          )
        ) AS altitude_rate_fps
      FROM with_duration_filtered
    ),

    with_features AS (
      SELECT
        *,
        CASE
          WHEN delta_t_sec IS NULL OR delta_t_sec = 0 THEN NULL
          ELSE 60.0 * SAFE_CAST(altitude_rate_fps AS FLOAT64)
        END AS d_alt_fpm
      FROM with_rates
    )

    SELECT
      Callsign,
      Generated_TS,
      Latitude,
      Longitude,
      Altitude,
      GroundSpeed,
      delta_t_sec,
      d_alt_fpm,
      flight_duration_sec,
      CASE
        WHEN GroundSpeed < 40 AND Altitude < 500 THEN "Ground"
        WHEN Altitude >= 25000 AND ABS(IFNULL(d_alt_fpm, 0)) < 300 THEN "Cruise"
        WHEN IFNULL(d_alt_fpm, 0) >= 300 THEN "Climb"
        WHEN IFNULL(d_alt_fpm, 0) <= -300 THEN "Descent"
        ELSE "Cruise"
      END AS Phase
    FROM with_features
    ORDER BY Callsign, Generated_TS
    {limit_clause}
    """
    
    if limit:
        print(f"\n📊 Executing BigQuery query (LIMIT {limit:,} rows)...")
    else:
        print("\n📊 Executing BigQuery query (fetching ALL data)...")
    
    job = client.query(query)
    df = job.to_dataframe()
    
    print(f"✅ Query completed. Retrieved {len(df):,} rows")
    print(f"\n📈 Phase distribution:")
    print(df['Phase'].value_counts())
    print(f"\n✈️  Total unique flights: {df['Callsign'].nunique()}")
    
    return df


def visualize_flight_profiles(df, num_callsigns=15, save_file=True, show_plot=True, seed=42):
    """
    Visualize altitude profiles for randomly selected flights.
    
    Args:
        df: DataFrame containing flight data with phases
        num_callsigns: Number of random flights to visualize
        save_file: Whether to save the figure to a file
        show_plot: Whether to display the plot in a popup window
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Randomly select callsigns
    unique_callsigns = df['Callsign'].unique()
    if len(unique_callsigns) < num_callsigns:
        print(f"⚠️  Warning: Only {len(unique_callsigns)} unique callsigns available")
        num_callsigns = len(unique_callsigns)
    
    selected_callsigns = random.sample(list(unique_callsigns), num_callsigns)
    print(f"\n🎯 Selected {num_callsigns} callsigns for visualization:")
    for i, cs in enumerate(selected_callsigns, 1):
        print(f"  {i}. {cs}")
    
    # Phase color mapping
    phases = df['Phase'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(phases)))
    phase_color_map = dict(zip(phases, colors))
    
    # Create subplots
    fig_height = 2.5 * num_callsigns
    fig, axes = plt.subplots(num_callsigns, 1, figsize=(16, fig_height))
    
    # Handle single subplot case
    if num_callsigns == 1:
        axes = [axes]
    
    # Plot each flight
    for idx, callsign in enumerate(selected_callsigns):
        ax = axes[idx]
        
        # Filter and sort flight data
        flight_data = df[df['Callsign'] == callsign].copy()
        flight_data = flight_data.sort_values('Generated_TS')
        
        # Create colored bar chart by phase
        for bar_idx, row in flight_data.iterrows():
            color = phase_color_map[row['Phase']]
            ax.bar(bar_idx, row['Altitude'], color=color, width=1.0, edgecolor='none')
        
        # Customize plot
        ax.set_title(f'Flight Profile - {callsign}', fontsize=18, fontweight='bold')
        ax.set_xlabel('Time Sequence (sorted by Generated_TS)', fontsize=15)
        ax.set_ylabel('Altitude', fontsize=15)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Format y-axis with k/M units
        ax.yaxis.set_major_formatter(FuncFormatter(format_altitude))
        ax.tick_params(axis='both', labelsize=13)
        
        # Add phase legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=phase_color_map[phase], label=phase)
            for phase in phases if phase in flight_data['Phase'].values
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=13)
    
    plt.tight_layout()
    
    # Save to file if requested
    if save_file:
        filename = f'flight_profiles_{num_callsigns}_callsigns.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f'\n💾 Saved: {filename}')
    
    # Show plot if requested
    if show_plot:
        print(f'\n🖼️  Displaying visualization popup...')
        plt.show()
    else:
        plt.close()


def main(
    credentials='/Users/sb/keys/iitp-class-team-2-473114-c86a1ab6eeba.json',
    project_id='iitp-class-team-2-473114',
    num_flights=15,
    no_show=False,
    no_save=False,
    limit=None,
    seed=42
):
    """
    Main function to orchestrate the flight phase classification and visualization.
    
    Args:
        credentials: Path to Google Cloud service account JSON key file
        project_id: Google Cloud project ID
        num_flights: Number of random flights to visualize
        no_show: Do not display plot popup (only save to file)
        no_save: Do not save plot to file (only display)
        limit: Limit number of rows to query from BigQuery (None for all data, e.g., 10000 for testing)
    """
    print("=" * 80)
    print("✈️  AIRCRAFT FLIGHT PHASE CLASSIFICATION AND VISUALIZATION")
    print("=" * 80)
    
    # Step 1: Verify credentials
    print("\n" + "=" * 80)
    print("STEP 1: VERIFYING CREDENTIALS")
    print("=" * 80)
    if not verify_credentials(credentials):
        return
    
    # Step 2: Create BigQuery client
    print("\n" + "=" * 80)
    print("STEP 2: CREATING BIGQUERY CLIENT")
    print("=" * 80)
    client = create_bigquery_client(project_id)
    
    # Step 3: Query and classify flights
    print("\n" + "=" * 80)
    print("STEP 3: QUERYING AND CLASSIFYING FLIGHT PHASES")
    print("=" * 80)
    df = query_and_classify_flights(client, limit=limit)
    
    # Step 4: Visualize flight profiles
    print("\n" + "=" * 80)
    print("STEP 4: VISUALIZING FLIGHT PROFILES")
    print("=" * 80)
    visualize_flight_profiles(
        df,
        num_callsigns=num_flights,
        save_file=not no_save,
        show_plot=not no_show,
        seed=seed
    )
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    fire.Fire(main)
