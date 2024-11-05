import React, { useState } from 'react';
import { Modal } from './Modal';

interface TooltipProps {
  children: React.ReactElement;
  content: React.ReactNode;
}

export const Tooltip: React.FC<TooltipProps> = ({ children, content }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <>
      <div 
        className="inline-block cursor-pointer"
        onClick={() => setIsModalOpen(true)}
      >
        {children}
      </div>

      <Modal 
        isOpen={isModalOpen} 
        onClose={() => setIsModalOpen(false)}
      >
        <div className="text-gray-800">
          {content}
        </div>
      </Modal>
    </>
  );
}; 