import React, { useEffect, useRef, useState } from 'react';

export default function ProgressiveMath({ children, threshold = 0.25, className = '' }) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (!ref.current || visible) return undefined;
    if (typeof IntersectionObserver === 'undefined') {
      setVisible(true);
      return undefined;
    }
    const observer = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          setVisible(true);
          observer.disconnect();
          return;
        }
      }
    }, { threshold });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [visible, threshold]);

  return (
    <div
      ref={ref}
      className={`transition-opacity duration-500 ease-out ${visible ? 'opacity-100' : 'opacity-0'} ${className}`}
    >
      {children}
    </div>
  );
}
