import argparse
from src.agent.graph import ASIAgent
import os

def main():
    parser = argparse.ArgumentParser(description="Autonomous Safety Investigator")
    parser.add_argument("--report", type=str, help="The safety report text to analyze")
    parser.add_argument("--file", type=str, help="Path to a file containing the report")
    
    args = parser.parse_args()
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("NOTICE: OPENAI_API_KEY environment variable not set. Running in MOCK mode.")
    
    report_text = ""
    if args.file:
        with open(args.file, "r") as f:
            report_text = f.read()
    elif args.report:
        report_text = args.report
    else:
        # Default case study from prompt
        report_text = """
        Location: SAN. Airplane: B737 MAX 8.
        Event: Descent.
        Narrative: Experiencing unstable approach and sink rate due to wake turbulence from preceding A321.
        """
        print("No input provided. Using default Case Study (SAN / B737 MAX 8) AS AN EXAMPLE.")
        
    try:
        agent = ASIAgent()
        result = agent.run(report_text)
        
        print("\n\n========== FINAL REPORT ==========\n")
        print(result.get("final_report"))
    except Exception as e:
        print(f"\nError running agent: {e}")
    
if __name__ == "__main__":
    main()
