import argparse
import time
from judgekit import create_judge, default_registry, generate_criteria_from_phrase


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge factory demo using JudgeKit.")
    parser.add_argument("--criteria", "-c", type=str, default="CodeReview", help="Preset name or phrase to create a judge.")
    parser.add_argument("--candidate", "-x", type=str, default="", help="Candidate text to evaluate.")
    parser.add_argument("--context", "-k", type=str, default="", help="Optional context string.")
    parser.add_argument("--description", "-d", type=str, default="Evaluate the candidate quality.", help="Evaluation description.")
    parser.add_argument("--recursive", "-r", type=int, default=0, help="If >0, create a meta-judge to assess the rubric.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose console hooks.")
    args = parser.parse_args()

    judge = create_judge(args.criteria, verbose=args.verbose)

    candidate = args.candidate or (
        "def add(a,b):return a+b # TODO: improve\n"
        "print(add(2,3))"
        if args.criteria.lower() == "codereview" else
        "Our product helps teams collaborate faster. It saves time and money. Learn more on our website."
    )
    description = args.description
    print(f"\n[demo] Evaluating with judge={args.criteria!r} ...")
    t0 = time.time()
    res1 = judge.evaluate(candidate=candidate, description=description, context=args.context)
    print(f"[demo] Score={res1.score:.3f} Passed={res1.passed} Took={(time.time()-t0)*1000:.0f}ms")
    print(f"[demo] Rationale: {res1.rationale}\n")

    # Optional meta-judge recursion: judge the rubric quality used to create the first judge
    if args.recursive > 0:
        # Obtain criteria text for the current judge
        reg = default_registry()
        crit = reg.get(args.criteria) or generate_criteria_from_phrase(args.criteria)
        rubric_text = crit.template
        meta_phrase = f"Rubric quality for: {args.criteria}"
        meta_judge = create_judge(meta_phrase, verbose=args.verbose)
        print(f"\n[demo] Meta-evaluating the rubric with judge={meta_phrase!r} ...")
        t1 = time.time()
        meta_res = meta_judge.evaluate(candidate=rubric_text, description="Evaluate the rubric quality.", context="")
        print(f"[demo] Meta Score={meta_res.score:.3f} Passed={meta_res.passed} Took={(time.time()-t1)*1000:.0f}ms")
        print(f"[demo] Meta Rationale: {meta_res.rationale}")


if __name__ == "__main__":
    main()


