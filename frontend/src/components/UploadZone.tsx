import React, { useState } from 'react';
import { Upload, Image, FileVideo } from 'lucide-react';
import { Card } from './ui/card';

interface UploadZoneProps {
  type: 'image' | 'video';
  onFileSelect: (file: File) => void;
}

export function UploadZone({ type, onFileSelect }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      setSelectedFile(files[0]);
      onFileSelect(files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setSelectedFile(files[0]);
      onFileSelect(files[0]);
    }
  };

  const Icon = type === 'image' ? Image : FileVideo;
  const accept = type === 'image' ? 'image/*' : 'video/*';

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
        isDragging
          ? 'border-primary bg-primary/10'
          : 'border-border bg-secondary/30 hover:border-primary/50'
      }`}
    >
      <Icon className={`w-12 h-12 mx-auto mb-4 ${isDragging ? 'text-primary' : 'text-muted-foreground'}`} />
      <p className="text-foreground mb-2">
        {selectedFile ? selectedFile.name : `Drag and drop ${type} file here`}
      </p>
      <p className="text-sm text-muted-foreground mb-4">or</p>
      <label className="inline-block">
        <input
          type="file"
          accept={accept}
          onChange={handleFileInput}
          className="hidden"
        />
        <span className="px-4 py-2 bg-primary text-primary-foreground rounded-lg cursor-pointer hover:bg-primary/90 transition-colors inline-block">
          Browse Files
        </span>
      </label>
      {selectedFile && (
        <p className="text-xs text-muted-foreground mt-4">
          File size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
        </p>
      )}
    </div>
  );
}
