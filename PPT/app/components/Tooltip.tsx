import React, { useState } from 'react';
import { Modal } from './Modal';
import { Info } from 'lucide-react';

interface TooltipProps {
  children: React.ReactElement;
  content: React.ReactNode;
}

export const Tooltip: React.FC<TooltipProps> = ({ children, content }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <>
      <div 
        className="inline-flex items-center gap-1 cursor-pointer"
        onClick={() => setIsModalOpen(true)}
      >
        {children}
        <Info className="w-4 h-4 text-blue-500" />
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