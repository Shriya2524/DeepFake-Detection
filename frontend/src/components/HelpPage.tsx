import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './ui/accordion';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Label } from './ui/label';
import { HelpCircle, Mail, MessageSquare } from 'lucide-react';
import { toast } from 'sonner@2.0.3';

export function HelpPage() {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    toast.success('Your message has been sent successfully!');
  };

  const faqs = [
    {
      question: 'How do I upload a file for analysis?',
      answer: 'Navigate to the Dashboard page and choose either the Image Analysis or Video Analysis section. You can drag and drop your file into the upload zone, or click "Browse Files" to select a file from your computer. Supported formats include JPEG, PNG for images, and MP4, AVI for videos.'
    },
    {
      question: 'What file formats are supported?',
      answer: 'For images, we support JPEG, PNG, and BMP formats. For videos, we support MP4, AVI, MOV, and MKV formats. Maximum file size is 100MB for images and 500MB for videos.'
    },
    {
      question: 'How long does the analysis take?',
      answer: 'Analysis time varies based on file size and type. Images typically take 5-15 seconds, while videos can take 30 seconds to 2 minutes depending on length and resolution. You\'ll see a progress bar during the analysis.'
    },
    {
      question: 'What does the confidence score mean?',
      answer: 'The confidence score represents how certain our AI model is about its prediction. A score of 95% means the model is 95% confident in its classification. Higher scores indicate stronger evidence of the predicted result.'
    },
    {
      question: 'How accurate is the detection system?',
      answer: 'Our system achieves an average accuracy of 96.8% across various deepfake detection benchmarks. However, accuracy can vary depending on the quality and sophistication of the deepfake. We continuously update our models to improve detection rates.'
    },
    {
      question: 'What is GAN Fingerprinting?',
      answer: 'GAN Fingerprinting is a technique that identifies unique patterns left by Generative Adversarial Networks (GANs) during the creation of synthetic media. Each GAN architecture leaves distinctive artifacts that can be detected and used to determine if media has been artificially generated.'
    },
    {
      question: 'What is Spatiotemporal Analysis?',
      answer: 'Spatiotemporal Analysis examines both spatial (within individual frames) and temporal (across multiple frames) features of videos. This helps identify inconsistencies in motion, lighting, and facial expressions that are common in deepfake videos but rare in authentic footage.'
    },
    {
      question: 'Can I download the analysis results?',
      answer: 'Yes! After completing an analysis, you can download a detailed PDF report that includes all metrics, detected anomalies, and confidence scores. Navigate to the Analysis page and click the "Download Report" button.'
    },
    {
      question: 'How can I export my analysis history?',
      answer: 'Go to the Results & Reports page where you can view all your previous analyses. You can export your analysis history as either a PDF or CSV file using the export buttons at the top of the results table.'
    },
    {
      question: 'Is my uploaded data stored?',
      answer: 'For this research prototype, uploaded files are processed in memory and not permanently stored on our servers. Analysis results are stored temporarily for your session. We take privacy and security seriously.'
    },
  ];

  return (
    <div className="max-w-4xl mx-auto">
      {/* Page Header */}
      <div className="mb-8 text-center">
        <div className="w-16 h-16 bg-primary rounded-2xl flex items-center justify-center mx-auto mb-4">
          <HelpCircle className="w-10 h-10 text-white" />
        </div>
        <h1 className="text-foreground mb-2">Help & Support</h1>
        <p className="text-muted-foreground">
          Find answers to common questions and get support
        </p>
      </div>

      {/* Getting Started */}
      <Card className="shadow-md mb-8 bg-card border-border">
        <CardHeader>
          <CardTitle className="text-foreground">Getting Started</CardTitle>
          <CardDescription>Quick guide to using the detection system</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-3">
            <div className="w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center flex-shrink-0">
              1
            </div>
            <div>
              <p className="text-foreground mb-1">Upload Your Media</p>
              <p className="text-sm text-muted-foreground">
                Go to the Dashboard and upload an image or video file you want to analyze
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <div className="w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center flex-shrink-0">
              2
            </div>
            <div>
              <p className="text-foreground mb-1">Start Analysis</p>
              <p className="text-sm text-muted-foreground">
                Click the "Analyze" button and wait for the AI to process your file
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <div className="w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center flex-shrink-0">
              3
            </div>
            <div>
              <p className="text-foreground mb-1">Review Results</p>
              <p className="text-sm text-muted-foreground">
                View the detection results, confidence scores, and detailed metrics
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <div className="w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center flex-shrink-0">
              4
            </div>
            <div>
              <p className="text-foreground mb-1">Export Reports</p>
              <p className="text-sm text-muted-foreground">
                Download detailed reports or view your analysis history in the Results page
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* FAQ Section */}
      <Card className="shadow-md mb-8 bg-card border-border">
        <CardHeader>
          <CardTitle className="text-foreground">Frequently Asked Questions</CardTitle>
          <CardDescription>Common questions about the detection system</CardDescription>
        </CardHeader>
        <CardContent>
          <Accordion type="single" collapsible className="w-full">
            {faqs.map((faq, index) => (
              <AccordionItem key={index} value={`item-${index}`}>
                <AccordionTrigger className="text-left text-foreground">
                  {faq.question}
                </AccordionTrigger>
                <AccordionContent className="text-muted-foreground">
                  {faq.answer}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </CardContent>
      </Card>

      {/* Contact Form */}
      <Card className="shadow-md bg-card border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-foreground">
            <MessageSquare className="w-5 h-5 text-primary" />
            Contact Support
          </CardTitle>
          <CardDescription>
            Can't find what you're looking for? Send us a message
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="name" className="text-foreground">Name</Label>
                <Input id="name" placeholder="Your name" required />
              </div>
              <div className="space-y-2">
                <Label htmlFor="email" className="text-foreground">Email</Label>
                <Input id="email" type="email" placeholder="your.email@example.com" required />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="subject" className="text-foreground">Subject</Label>
              <Input id="subject" placeholder="Brief description of your issue" required />
            </div>
            <div className="space-y-2">
              <Label htmlFor="message" className="text-foreground">Message</Label>
              <Textarea
                id="message"
                placeholder="Describe your issue or question in detail..."
                className="min-h-[120px]"
                required
              />
            </div>
            <Button type="submit" className="w-full bg-primary hover:bg-primary/90">
              <Mail className="w-4 h-4 mr-2" />
              Send Message
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
