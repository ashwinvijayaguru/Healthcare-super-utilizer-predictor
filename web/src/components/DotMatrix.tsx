/**
 * One hundred squares, N filled.
 *
 * People reason about frequencies far better than percentages, and the empty
 * squares do work that a number cannot: they make "most people in this position
 * do not end up in hospital" visible rather than merely asserted. That matters
 * on a screen a patient may be looking at.
 */
export default function DotMatrix({ filled }: { filled: number }) {
  const count = Math.max(0, Math.min(100, Math.round(filled)));

  return (
    <div>
      <div className="matrix" role="img" aria-label={`${count} of 100 squares filled`}>
        {Array.from({ length: 100 }, (_, i) => (
          <i
            key={i}
            data-on={i < count}
            style={{ animationDelay: i < count ? `${i * 6}ms` : undefined }}
          />
        ))}
      </div>
      <p className="matrix-caption">
        Of 100 members with a similar history, about <strong>{count}</strong> meet the
        outcome within a year. <em>The other {100 - count} do not.</em>
      </p>
    </div>
  );
}
