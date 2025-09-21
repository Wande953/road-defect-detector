# export_data.py
import json
import csv
from datetime import datetime, timedelta


def export_reports(start_date, end_date):
    """Export maintenance reports to CSV"""
    with open('data/reports/maintenance_reports.csv', 'r') as f:
        reader = csv.DictReader(f)
        reports = [row for row in reader if start_date <= row['Date'] <= end_date]

    export_filename = f"reports_export_{datetime.now().strftime('%Y%m%d')}.csv"
    with open(export_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(reports)

    return export_filename


def generate_weekly_report():
    """Generate weekly summary report"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    reports = export_reports(start_date, end_date)
    print(f"Weekly report generated: {reports}")