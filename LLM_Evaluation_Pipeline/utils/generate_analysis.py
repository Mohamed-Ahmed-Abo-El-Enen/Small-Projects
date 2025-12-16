from utils.analysis.error_analysis import ErrorAnalyzer


if __name__ == '__main__':
    analyzer = ErrorAnalyzer()
    report = analyzer.generate_report()
    print(report)

    with open('results/error_analysis_report.txt', 'w') as f:
        f.write(report)

    analyzer.create_visualizations()