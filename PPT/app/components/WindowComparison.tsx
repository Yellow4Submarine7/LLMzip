import React, { useEffect, useState } from 'react';

const WindowComparison = () => {
  const [step, setStep] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);

  // Example text for demonstrating parallel processing
  const tokens1 = ["The", "quick", "brown", "fox", "jumps"];
  const tokens2 = ["over", "the", "lazy", "dog", "now"];
  
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(() => {
        setStep((prev) => (prev + 1) % 5 || 1); // Modified to 5 as each block has 5 tokens
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  const renderLLMZip = () => {
    const currentIndex = step - 1;
    const windowSize = 4;
    const start = Math.max(0, currentIndex);

    return (
      <div className="flex flex-col space-y-2">
        <div className="flex space-x-1">
          {tokens1.concat(tokens2).map((token, idx) => (
            <div
              key={idx}
              className={`px-2 py-1 rounded ${
                idx >= start && idx < start + windowSize
                  ? 'bg-red-200 border-2 border-red-500'
                  : 'bg-gray-100'
              }`}
            >
              {token}
            </div>
          ))}
        </div>
        <div className="text-sm text-red-600">
          Fixed window: Predicting the next token requires processing the complete context of {windowSize} tokens
        </div>
      </div>
    );
  };

  const renderFineZip = () => {
    const currentIndex = step - 1;
    return (
      <div className="flex flex-col space-y-2">
        <div className="flex space-x-1">
          {tokens1.map((token, idx) => (
            <div
              key={idx}
              className={`px-2 py-1 rounded ${
                idx <= currentIndex
                  ? 'bg-green-200 border-2 border-green-500'
                  : 'bg-gray-100'
              }`}
            >
              {token}
            </div>
          ))}
        </div>
        <div className="flex space-x-1">
          {tokens2.map((token, idx) => (
            <div
              key={idx}
              className={`px-2 py-1 rounded ${
                idx <= currentIndex
                  ? 'bg-green-200 border-2 border-green-500'
                  : 'bg-gray-100'
              }`}
            >
              {token}
            </div>
          ))}
        </div>
        <div className="text-sm text-green-600">
          Dynamic window: Context grows gradually, enabling parallel processing of multiple blocks
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-8">
      <div className="flex justify-center space-x-4">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          {isPlaying ? 'Pause' : 'Start'} Animation
        </button>
        <button
          onClick={() => setStep((prev) => (prev % 5) + 1)}
          className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
        >
          Next Step
        </button>
      </div>
      <div className="space-y-8">
        <div>
          <h3 className="text-xl font-semibold mb-4">LLMZip: Fixed Window</h3>
          {renderLLMZip()}
        </div>
        <div>
          <h3 className="text-xl font-semibold mb-4">FineZip: Dynamic Window</h3>
          {renderFineZip()}
        </div>
      </div>
    </div>
  );
};

export default WindowComparison; 