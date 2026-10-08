import { useNavigate, useSearchParams } from "react-router-dom";

export const ORDER = ["profile", "lifestyle", "health", "review"];

// Back / Continue navigation. When a step was opened from the Review page (?edit=1),
// Continue returns straight to Review so editing is one click.
export default function useStepNav(current) {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const editing = params.get("edit") === "1";
  const i = ORDER.indexOf(current);
  return {
    editing,
    next: () => navigate(editing ? "/screening/review" : `/screening/${ORDER[i + 1]}`),
    back: () => navigate(editing ? "/screening/review" : i === 0 ? "/" : `/screening/${ORDER[i - 1]}`),
  };
}
