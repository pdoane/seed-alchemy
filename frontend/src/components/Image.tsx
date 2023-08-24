import { useEffect, useState } from "react";

interface ImageProps {
  src: string;
  className: string;
}

export const Image = ({ src, className }: ImageProps) => {
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    setHasError(false);
  }, [src]);

  function handleError(): void {
    setHasError(true);
  }

  if (hasError) return <></>;
  return <img src={src} className={className} onError={handleError} />;
};
